"""
RT Agent Evaluator
评测agent每一步re-planning的质量，包括safety、progress和cost三个维度
"""
import numpy as np
from typing import List, Tuple, Dict
from simworld.utils.vector import Vector
from simworld.utils.logger import Logger
from base.rt_action_space import MOVE_TO, TURN_AROUND, WAIT


class RTEvaluator:
    """评测agent每一步的re-planning质量"""
    
    def __init__(self, communicator, agent):
        """
        初始化评测器
        
        Args:
            communicator: RT通信器，用于获取环境信息
            agent: RT Agent实例
        """
        self.communicator = communicator
        self.agent = agent
        self.logger = Logger.get_logger('RTEvaluator')
        
        # Get static obstacles from agent
        self.static_obstacles = agent.static_obstacles if hasattr(agent, 'static_obstacles') else []
        self.dynamic_obstacles = dict(
            getattr(agent, 'dynamic_obstacles', {}) or {}
        )
        self.dynamic_movement_threshold_cm = 10.0
        self._latest_dynamic_states = {}
        self._last_context_dynamic_states = {}
        self._latest_dynamic_z_velocities = {}
        self._dynamic_actor_telemetry = {
            actor_id: {
                'actor_id': actor_id,
                'kind': metadata.get('kind', 'dynamic'),
                'initial_location_cm': list(
                    metadata.get('initial_location_cm', [0.0, 0.0, 0.0])
                ),
                'baseline_live_location_cm': None,
                'last_location_cm': None,
                'max_displacement_cm': 0.0,
                'max_planar_displacement_cm': 0.0,
                'sample_count': 0,
                'observed_moving': False,
            }
            for actor_id, metadata in self.dynamic_obstacles.items()
        }

        # 评分权重配置
        self.safety_weight = 1.0
        self.progress_weight = 1.0
        self.cost_weight = 1.0
        
        # Cost惩罚值配置
        self.cost_penalties = {
            'MOVE': 0.0,           # 前进没有惩罚
            'TURN': -0.1,          # 转弯轻微惩罚
            'WAIT': -0.5           # wait惩罚最大
        }
        
        # Safety距离阈值配置（单位：cm）
        self.safety_thresholds = {
            'danger': 70,    # < 70cm 非常危险
            'unsafe': 100,    # < 100cm 不安全
            'safe': 200       # > 200cm 安全
        }
        self.route_corridor_limit = 220.0
        self.route_correction_margin = 50.0
        
        # 历史记录
        self.step_records = []
        self.last_preview_npc_positions = None
        self.last_preview_sim_time = None
        self._last_context_npc_positions = {}
        self._last_context_npc_velocities = {}

    def build_action_safety_context(
        self,
        candidate_waypoints: List[Vector],
        expected_latency: float = 0.0,
        fixed_action_duration: float | None = None,
        npc_velocities_override: Dict[str, Vector] | None = None,
    ) -> Dict:
        """
        Build a pre-action safety/progress summary for the VLM prompt.

        This is the online counterpart to the end-of-step counterfactual evaluator:
        it uses current NPC locations plus a velocity estimate from the previous
        planning step, then scores every action for the expected execution window.
        """
        expected_latency = max(0.0, float(expected_latency or 0.0))
        fixed_action_duration = (
            max(0.0, float(fixed_action_duration))
            if fixed_action_duration is not None else
            None
        )
        agent_position = self.agent.position
        npc_positions = self._get_npc_positions()
        npc_velocities = (
            dict(npc_velocities_override)
            if npc_velocities_override is not None
            else self._estimate_npc_velocities(npc_positions)
        )
        self._last_context_npc_positions = npc_positions
        self._last_context_npc_velocities = npc_velocities
        self._last_context_dynamic_states = dict(self._latest_dynamic_states)

        path_distance_start = self._calculate_path_distance(
            agent_position,
            self.agent.shortest_path
        )

        candidate_scores = {}

        for i, waypoint in enumerate(candidate_waypoints):
            action_id = str(i + 1)
            distance = agent_position.distance(waypoint)
            duration = fixed_action_duration if fixed_action_duration is not None else round(distance / max(self.agent.speed, 1), 2) + 1.0
            min_distance, closest_obstacles = self._calculate_predicted_min_distance_details(
                agent_position,
                waypoint,
                npc_positions,
                npc_velocities,
                expected_latency,
                duration
            )
            route_feasible, route_details = self._assess_route_corridor(agent_position, waypoint)
            if not route_feasible:
                min_distance = min(min_distance, 0.0)
                closest_obstacles.append(route_details)
                closest_obstacles.sort(key=lambda item: item['distance_cm'])
                closest_obstacles = closest_obstacles[:3]
            safety = self._score_from_min_distance(min_distance)
            hypothetical_path_distance = self._calculate_path_distance(
                waypoint,
                self.agent.shortest_path
            )
            progress = self._calculate_progress_score(path_distance_start, hypothetical_path_distance)
            cost = self._calculate_cost_score('MOVE')
            total = (
                self.safety_weight * safety +
                self.progress_weight * progress +
                self.cost_weight * cost
            )
            candidate_scores[action_id] = {
                'action_type': MOVE_TO,
                'action_param': action_id,
                'duration_s': duration,
                'latency_s': expected_latency,
                'min_clearance_cm': min_distance,
                'safety_score': safety,
                'progress_score': progress,
                'cost_score': cost,
                'total_score': total,
                'closest_obstacles': closest_obstacles,
                'is_safe': self._is_safe_clearance(min_distance, safety),
                'is_progressing': progress > 0.0,
            }

        for turn_id in ['L30', 'L60', 'L90', 'R30', 'R60', 'R90']:
            duration = fixed_action_duration if fixed_action_duration is not None else 1.0
            min_distance, closest_obstacles = self._calculate_predicted_min_distance_details(
                agent_position,
                agent_position,
                npc_positions,
                npc_velocities,
                expected_latency,
                duration
            )
            safety = self._score_from_min_distance(min_distance)
            cost = self._calculate_cost_score('TURN')
            candidate_scores[turn_id] = {
                'action_type': TURN_AROUND,
                'action_param': turn_id,
                'duration_s': duration,
                'latency_s': expected_latency,
                'min_clearance_cm': min_distance,
                'safety_score': safety,
                'progress_score': 0.0,
                'cost_score': cost,
                'total_score': self.safety_weight * safety + self.cost_weight * cost,
                'closest_obstacles': closest_obstacles,
                'is_safe': self._is_safe_clearance(min_distance, safety),
                'is_progressing': False,
            }

        for wait_id in ['WAIT_1', 'WAIT_2', 'WAIT_3']:
            duration = fixed_action_duration if fixed_action_duration is not None else float(wait_id.split('_')[1])
            min_distance, closest_obstacles = self._calculate_predicted_min_distance_details(
                agent_position,
                agent_position,
                npc_positions,
                npc_velocities,
                expected_latency,
                duration
            )
            safety = self._score_from_min_distance(min_distance)
            cost = self._calculate_cost_score('WAIT') * duration
            candidate_scores[wait_id] = {
                'action_type': WAIT,
                'action_param': wait_id.split('_')[1],
                'duration_s': duration,
                'latency_s': expected_latency,
                'min_clearance_cm': min_distance,
                'safety_score': safety,
                'progress_score': 0.0,
                'cost_score': cost,
                'total_score': self.safety_weight * safety + self.cost_weight * cost,
                'closest_obstacles': closest_obstacles,
                'is_safe': self._is_safe_clearance(min_distance, safety),
                'is_progressing': False,
            }

        best_action_id, best_action = max(
            candidate_scores.items(),
            key=lambda item: item[1]['total_score']
        )
        safe_progress_actions = {
            action_id: score
            for action_id, score in candidate_scores.items()
            if score['is_safe'] and score['is_progressing']
        }
        safe_hold_actions = {
            action_id: score
            for action_id, score in candidate_scores.items()
            if score['is_safe'] and not score['is_progressing']
        }
        if safe_progress_actions:
            best_safe_progress_id, best_safe_progress = max(
                safe_progress_actions.items(),
                key=lambda item: item[1]['total_score']
            )
        else:
            best_safe_progress_id, best_safe_progress = None, None

        self.last_preview_npc_positions = npc_positions
        self.last_preview_sim_time = getattr(self.agent, 'sim_time_elapsed', None)

        return {
            'expected_latency_s': expected_latency,
            'num_dynamic_obstacles': len(npc_positions),
            'num_static_obstacles': len(self.static_obstacles),
            'safe_path_action_exists': bool(safe_progress_actions),
            'safe_hold_action_exists': bool(safe_hold_actions),
            'best_action': best_action_id,
            'best_action_score': best_action,
            'best_safe_progress_action': best_safe_progress_id,
            'best_safe_progress_score': best_safe_progress,
            'candidate_scores': candidate_scores,
        }

    def build_latency_safety_sweep(
        self,
        candidate_waypoints: List[Vector],
        latency_budgets: List[float] | Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0),
        fixed_action_duration: float | None = None,
    ) -> Dict:
        """Build a latency-indexed safety curve for every candidate action.

        The returned sweep is a structured bootstrap signal for the
        latency-aware safety-distribution critic.  Each entry contains the
        existing evaluator's predicted clearance, collision-safety score, and
        progress at one latency budget.

        This is a model-based preview, not yet an empirical probability
        distribution.  To obtain calibrated probabilities, replay the same
        observation/action across multiple seeds and replace the preview
        values with observed survival labels before training the critic.
        """
        budgets = sorted({max(0.0, float(value)) for value in latency_budgets})
        previews = []
        curves: Dict[str, Dict[str, List]] = {}
        for latency in budgets:
            preview = self.build_action_safety_context(
                candidate_waypoints,
                expected_latency=latency,
                fixed_action_duration=fixed_action_duration,
            )
            previews.append(preview)
            for action_id, score in preview.get('candidate_scores', {}).items():
                entry = curves.setdefault(action_id, {
                    'action_id': action_id,
                    'action_type': score.get('action_type'),
                    'action_param': score.get('action_param'),
                    'latency_s': [],
                    'min_clearance_cm': [],
                    'safety_score': [],
                    'progress_score': [],
                    'is_safe': [],
                    'is_progressing': [],
                    'closest_obstacles': [],
                })
                entry['latency_s'].append(latency)
                entry['min_clearance_cm'].append(score.get('min_clearance_cm'))
                entry['safety_score'].append(score.get('safety_score'))
                entry['progress_score'].append(score.get('progress_score'))
                entry['is_safe'].append(bool(score.get('is_safe')))
                entry['is_progressing'].append(bool(score.get('is_progressing')))
                entry['closest_obstacles'].append(score.get('closest_obstacles') or [])

        return {
            'latency_budgets_s': budgets,
            'previews': previews,
            'candidate_curves': curves,
        }

    def build_latency_safety_distribution_preview(
        self,
        candidate_waypoints: List[Vector],
        latency_budgets: List[float] | Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0),
        time_bins_s: List[float] | Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0),
        action_types: Tuple[str, ...] = (MOVE_TO,),
        npc_velocities_override: Dict[str, Vector] | None = None,
    ) -> Dict:
        """Build a structured safety-curve preview for each latency budget.

        For every ``latency`` and future ``time_bin``, the existing geometric
        evaluator is queried over that horizon.  The output is intended for
        bootstrapping/debugging the latency-aware data pipeline.  It is not an
        empirical probability distribution: use repeated UE action replays to
        populate observed collision-free survival labels.
        """
        latencies = sorted({max(0.0, float(value)) for value in latency_budgets})
        horizons = sorted({max(0.0, float(value)) for value in time_bins_s})
        curves: Dict[str, Dict] = {}

        for latency in latencies:
            for horizon in horizons:
                preview = self.build_action_safety_context(
                    candidate_waypoints,
                    expected_latency=latency,
                    fixed_action_duration=horizon,
                    npc_velocities_override=npc_velocities_override,
                )
                for action_id, score in preview.get('candidate_scores', {}).items():
                    if score.get('action_type') not in action_types:
                        continue
                    action_curve = curves.setdefault(action_id, {
                        'action_id': action_id,
                        'action_type': score.get('action_type'),
                        'action_param': score.get('action_param'),
                        'duration_s': score.get('duration_s'),
                        'latency_s': list(latencies),
                        'time_bins_s': list(horizons),
                        'min_clearance_cm': [[] for _ in latencies],
                        'safety_score': [[] for _ in latencies],
                        'progress_score': [[] for _ in latencies],
                        'is_safe': [[] for _ in latencies],
                        'is_progressing': [[] for _ in latencies],
                        'closest_obstacles': [[] for _ in latencies],
                    })
                    index = latencies.index(latency)
                    action_curve['min_clearance_cm'][index].append(score.get('min_clearance_cm'))
                    action_curve['safety_score'][index].append(score.get('safety_score'))
                    action_curve['progress_score'][index].append(score.get('progress_score'))
                    action_curve['is_safe'][index].append(bool(score.get('is_safe')))
                    action_curve['is_progressing'][index].append(bool(score.get('is_progressing')))
                    action_curve['closest_obstacles'][index].append(score.get('closest_obstacles', []))

        return {
            'latency_budgets_s': latencies,
            'time_bins_s': horizons,
            'label_source': 'rtevaluator_geometric_preview',
            'curves': curves,
        }

    def build_safe_trajectory_plan(
        self,
        expected_latency: float = 0.0,
        max_steps: int = 40,
        step_length: float = 350.0,
        target_tolerance: float = 250.0
    ) -> Dict:
        """
        Compute an internal location-based safe trajectory along the remaining
        route. This is logged for analysis and is not intended for direct prompt
        injection into the VLM.
        """
        expected_latency = max(0.0, float(expected_latency or 0.0))
        max_steps = max(1, int(max_steps))
        step_length = max(50.0, float(step_length))

        npc_positions = self._last_context_npc_positions or self._get_npc_positions()
        npc_velocities = self._last_context_npc_velocities or self._estimate_npc_velocities(npc_positions)

        route = self._remaining_route_polyline()
        if len(route) < 2:
            return {
                'feasible': True,
                'reaches_target': True,
                'reason': 'already_at_or_near_target',
                'expected_latency_s': expected_latency,
                'steps': [],
                'total_duration_s': 0.0,
                'reasonable_step_limit': 0,
            }

        route_distance = self._polyline_length(route)
        reasonable_step_limit = min(
            max_steps,
            max(8, int(route_distance / max(step_length * 0.7, 1.0)) + 8)
        )

        current = route[0]
        target = route[-1]
        route_index = 0
        cumulative_time = 0.0
        planned_steps = []
        waits_inserted = 0
        reason = 'target_reached'

        for plan_step in range(max_steps):
            if current.distance(target) <= target_tolerance:
                break

            next_point, route_index = self._advance_along_polyline(
                current,
                route,
                route_index,
                step_length
            )
            duration = round(current.distance(next_point) / max(self.agent.speed, 1), 2) + 1.0
            min_distance, closest_obstacles = self._calculate_predicted_min_distance_details(
                current,
                next_point,
                npc_positions,
                npc_velocities,
                expected_latency + cumulative_time,
                duration
            )
            route_feasible, route_details = self._assess_route_corridor(current, next_point)
            if not route_feasible:
                min_distance = min(min_distance, 0.0)
                closest_obstacles.append(route_details)
                closest_obstacles.sort(key=lambda item: item['distance_cm'])
                closest_obstacles = closest_obstacles[:3]

            safety = self._score_from_min_distance(min_distance)
            if self._is_safe_clearance(min_distance, safety):
                cumulative_time += duration
                planned_steps.append(
                    self._trajectory_step_record(
                        plan_step,
                        MOVE_TO,
                        None,
                        current,
                        next_point,
                        duration,
                        cumulative_time,
                        min_distance,
                        closest_obstacles,
                    )
                )
                current = next_point
                continue

            wait_record = self._find_safe_wait_for_trajectory(
                len(planned_steps),
                current,
                npc_positions,
                npc_velocities,
                expected_latency + cumulative_time,
                cumulative_time
            )
            if wait_record is None:
                reason = 'blocked_by_predicted_collision'
                break

            waits_inserted += 1
            cumulative_time = wait_record['cumulative_time_s']
            planned_steps.append(wait_record)
        else:
            reason = 'max_steps_exhausted'

        reaches_target = current.distance(target) <= target_tolerance
        feasible = reaches_target and len(planned_steps) <= reasonable_step_limit
        if reaches_target and not feasible:
            reason = 'exceeds_reasonable_step_limit'

        return {
            'feasible': feasible,
            'reaches_target': reaches_target,
            'reason': reason if not feasible else 'target_reached',
            'expected_latency_s': expected_latency,
            'start': self._point_dict(route[0]),
            'target': self._point_dict(target),
            'target_tolerance_cm': target_tolerance,
            'step_length_cm': step_length,
            'total_route_distance_cm': route_distance,
            'total_duration_s': round(cumulative_time, 2),
            'num_steps': len(planned_steps),
            'num_waits_inserted': waits_inserted,
            'reasonable_step_limit': reasonable_step_limit,
            'steps': planned_steps,
        }

    def format_safe_trajectory_plan(self, plan: Dict) -> str:
        """Format the internal safe-trajectory plan for per-step logs."""
        if not plan:
            return "No internal safe trajectory plan is available."

        lines = [
            "Internal computed safe trajectory (not shown to VLM):",
            (
                f"- feasible={plan.get('feasible')}, reaches_target={plan.get('reaches_target')}, "
                f"reason={plan.get('reason')}"
            ),
            (
                f"- expected_latency={plan.get('expected_latency_s', 0.0):.2f}s, "
                f"steps={plan.get('num_steps', 0)}/{plan.get('reasonable_step_limit', 0)}, "
                f"duration={plan.get('total_duration_s', 0.0):.2f}s"
            ),
        ]
        for step in plan.get('steps', [])[:12]:
            closest = step.get('closest_obstacle') or {}
            closest_desc = "none"
            if closest:
                closest_desc = (
                    f"{closest.get('kind')}:{closest.get('id')} "
                    f"at {self._format_distance(closest.get('distance_cm', float('inf')))}"
                )
            lines.append(
                f"  {step['step']} | {step['action_type']}:{step.get('action_param')} | "
                f"{step['duration_s']:.1f}s | clearance={self._format_distance(step['min_clearance_cm'])} | "
                f"to=({step['end']['x']:.1f}, {step['end']['y']:.1f}) | closest={closest_desc}"
            )
        remaining = len(plan.get('steps', [])) - 12
        if remaining > 0:
            lines.append(f"  ... {remaining} more planned steps")
        return "\n".join(lines)

    def format_action_safety_context(self, context: Dict) -> str:
        """Format the online safety preview for compact insertion into the prompt."""
        if not context:
            return "No safety preview is available; rely on visual perception and choose conservatively."

        candidate_scores = context.get('candidate_scores', {})
        lines = [
            "Online safety/progress preview from location data:",
            (
                f"- Expected reasoning latency before execution: "
                f"{context.get('expected_latency_s', 0.0):.2f}s."
            ),
            (
                f"- Obstacles considered: {context.get('num_dynamic_obstacles', 0)} dynamic, "
                f"{context.get('num_static_obstacles', 0)} static."
            ),
            (
                f"- Barrier-feasible progressing path action exists: "
                f"{'yes' if context.get('safe_path_action_exists') else 'no'}."
            ),
        ]

        best_safe_progress_id = context.get('best_safe_progress_action')
        if best_safe_progress_id:
            score = context['best_safe_progress_score']
            lines.append(
                "- Best safe progress action: "
                f"{best_safe_progress_id} "
                f"(safety={score['safety_score']:.2f}, progress={score['progress_score']:.2f}, "
                f"total={score['total_score']:.2f}, min_clearance={self._format_distance(score['min_clearance_cm'])})."
            )
        else:
            lines.append(
                "- Best safe progress action: none at the configured barrier threshold; "
                "compare the best cautious progress action against turn/wait instead of freezing."
            )

        best_action_id = context.get('best_action')
        best_action_score = context.get('best_action_score')
        if best_action_id and best_action_score:
            lines.append(
                "- Best achievable safety-progress-cost action overall: "
                f"{best_action_id} "
                f"(safety={best_action_score['safety_score']:.2f}, "
                f"progress={best_action_score['progress_score']:.2f}, "
                f"cost={best_action_score['cost_score']:.2f}, "
                f"total={best_action_score['total_score']:.2f})."
            )

        lines.append(
            "- Candidate table: id | type:param | duration | min_clearance | safety | progress | total | closest obstacle"
        )
        for action_id in self._ordered_action_ids():
            if action_id not in candidate_scores:
                continue
            score = candidate_scores[action_id]
            closest = score.get('closest_obstacles') or []
            closest_desc = "none"
            if closest:
                first = closest[0]
                closest_desc = (
                    f"{first['kind']}:{first['id']} "
                    f"at {self._format_distance(first['distance_cm'])}"
                )
            lines.append(
                f"  {action_id} | {score['action_type']}:{score['action_param']} | "
                f"{score['duration_s']:.1f}s | {self._format_distance(score['min_clearance_cm'])} | "
                f"{score['safety_score']:.2f} | {score['progress_score']:.2f} | "
                f"{score['total_score']:.2f} | {closest_desc}"
            )

        return "\n".join(lines)
        
    def start_step_evaluation(self, candidate_waypoints: List[Vector]) -> Dict:
        """
        开始一步评测，只记录初始状态，不进行评估
        
        Args:
            candidate_waypoints: 5个候选waypoints
        
        Returns:
            initial_state: 包含agent和NPC的初始位置信息
        """
        # 获取agent当前位置
        agent_position = self.agent.position
        
        # 获取所有NPC的位置（起始位置）
        npc_positions = self._get_npc_positions()
        
        # 计算到目标的路径距离
        path_distance = self._calculate_path_distance(
            self.agent.position, 
            self.agent.shortest_path
        )
        
        initial_state = {
            'agent_position': agent_position,
            'npc_positions': npc_positions,
            'dynamic_actor_states': dict(self._latest_dynamic_states),
            'path_distance': path_distance,
            'step_num': self.agent.step_num,
            'candidate_waypoints': candidate_waypoints  # 保存候选waypoints用于end时评估
        }
        
        return initial_state
    
    def end_step_evaluation(
        self,
        initial_state: Dict,
        action_type: str,
        action_choice: str,
        observed_safety_events: Dict | None = None,
    ) -> Dict:
        """
        结束一步评测，使用真实的NPC位置进行反事实评估，计算实际action分数并对比最优选择
        
        Args:
            initial_state: 从start_step_evaluation返回的初始状态
            action_type: 动作类型 ('MOVE', 'TURN', 'WAIT')
            action_choice: 实际选择的action ('1'-'7' for move, 'L30'/'R60' etc for turn, 'WAIT_1'/'WAIT_2'/'WAIT_3' for wait)
            
        Returns:
            scores: 包含safety、progress、cost和total分数的字典
        """
        # 获取agent结束位置（真实执行后的位置）
        agent_end_position = self.agent.position
        
        # 获取所有NPC的结束位置（Ground Truth）
        npc_end_positions = self._get_npc_positions()
        dynamic_end_states = dict(self._latest_dynamic_states)
        
        # 计算当前路径距离
        path_distance_now = self._calculate_path_distance(
            self.agent.position,
            self.agent.shortest_path
        )
        
        # 1. 计算实际执行动作的分数
        # A stationary agent can still be struck by a moving pedestrian,
        # movable prop, or falling object, so TURN/WAIT use the same relative
        # trajectory calculation as MOVE.
        actual_safety_score = self._calculate_safety_score(
            initial_state['agent_position'],
            agent_end_position,
            initial_state['npc_positions'],
            npc_end_positions,
            initial_state.get('dynamic_actor_states'),
            dynamic_end_states,
        )
        observed_safety_events = dict(observed_safety_events or {})
        observed_unsafe_event = any(
            int(observed_safety_events.get(key, 0) or 0) > 0
            for key in (
                'human',
                'object',
                'building',
                'vehicle',
                'touched_road',
                'fall',
                'oil',
                'water',
            )
        )
        # UE collision/overlap counters are authoritative observations.  The
        # endpoint-trajectory approximation can miss a contact when an actor
        # is pushed away before the end snapshot, so a realized unsafe event
        # must never retain a positive safety score.
        if observed_unsafe_event:
            actual_safety_score = 0.0
        
        actual_progress_score = self._calculate_progress_score(
            initial_state['path_distance'],
            path_distance_now
        )
        
        actual_cost_score = self._calculate_cost_score(action_type)
        
        actual_total_score = (
            self.safety_weight * actual_safety_score +
            self.progress_weight * actual_progress_score +
            self.cost_weight * actual_cost_score
        )
        
        # 2. 使用真实的NPC位置评估所有候选action（反事实评估）
        candidate_scores = self._evaluate_all_candidates(
            initial_state['agent_position'],
            initial_state['npc_positions'],
            npc_end_positions,  # 使用真实的NPC结束位置
            initial_state['path_distance'],
            initial_state['candidate_waypoints'],
            initial_state.get('dynamic_actor_states'),
            dynamic_end_states,
        )
        
        # 3. 找出最优action
        best_action = max(candidate_scores.items(), key=lambda x: x[1]['total_score'])
        best_action_id = best_action[0]
        best_score = best_action[1]['total_score']
        
        # 4. 在同一个名义候选空间中计算决策质量，并单独记录执行偏差
        chosen_candidate = candidate_scores.get(action_choice) if action_choice else None
        chosen_candidate_score = (
            chosen_candidate['total_score'] if chosen_candidate is not None else None
        )
        score_gap = (
            max(0.0, best_score - chosen_candidate_score)
            if chosen_candidate_score is not None
            else 0.0
        )
        # ``score_gap`` can be a NumPy scalar because candidate evaluation uses
        # NumPy geometry.  Convert the comparison to a native bool so the
        # aggregate rollout report remains JSON serializable.
        is_optimal = bool(
            chosen_candidate_score is not None and score_gap <= 1e-9
        )
        realization_score_delta = (
            actual_total_score - chosen_candidate_score
            if chosen_candidate_score is not None
            else None
        )
        
        scores = {
            'step_num': initial_state['step_num'],
            'safety_score': actual_safety_score,
            'progress_score': actual_progress_score,
            'cost_score': actual_cost_score,
            'total_score': actual_total_score,
            'action_type': action_type,
            'action_choice': action_choice,
            'min_npc_distance': self._min_distance if hasattr(self, '_min_distance') else None,
            'path_progress': initial_state['path_distance'] - path_distance_now,
            'best_action': best_action_id,
            'best_score': best_score,
            'chosen_candidate_score': chosen_candidate_score,
            'is_optimal': is_optimal,
            'score_gap': score_gap,
            'realization_score_delta': realization_score_delta,
            'all_candidates': candidate_scores,
            'observed_unsafe_event': observed_unsafe_event,
            'observed_safety_events': observed_safety_events,
        }
        
        self.step_records.append(scores)
        
        # self.logger.info(
        #     f'Step {initial_state["step_num"]}: '
        #     f'Choice={action_choice}, Best={best_action_id}, '
        #     f'Score={actual_total_score:.2f}, BestScore={best_score:.2f}, '
        #     f'Optimal={is_optimal}, Gap={score_gap:.2f}'
        # )
        
        return scores
    
    def _evaluate_all_candidates(
        self,
        agent_start_position: Vector,
        npc_start_positions: Dict[str, Vector],
        npc_end_positions: Dict[str, Vector],
        path_distance_start: float,
        waypoints: List[Vector],
        dynamic_start_states: Dict | None = None,
        dynamic_end_states: Dict | None = None,
    ) -> Dict[str, Dict]:
        """
        评估所有候选action的分数（反事实评估）
        使用真实的NPC结束位置来评估如果选择了其他action会怎样
        
        Args:
            agent_start_position: agent起始位置
            npc_start_positions: NPC起始位置
            npc_end_positions: NPC结束位置（Ground Truth）
            path_distance_start: 起始时到目标的路径距离
            waypoints: 5个候选waypoints
            
        Returns:
            candidate_scores: 字典 {action_id: {safety, progress, cost, total}}
        """
        candidate_scores = {}
        
        # 评估5个waypoint选择 (1-5)
        for i, waypoint in enumerate(waypoints):
            action_id = str(i + 1)
            # 假设选择这个waypoint，agent会到达的位置
            hypothetical_agent_end = waypoint
            
            # 使用真实的NPC结束位置计算safety分数
            safety = self._calculate_safety_score(
                agent_start_position, 
                hypothetical_agent_end,
                npc_start_positions, 
                npc_end_positions,  # 使用真实的NPC位置
                dynamic_start_states,
                dynamic_end_states,
            )
            
            # 计算progress分数
            # 假设到达这个waypoint后的路径距离
            hypothetical_path_distance = self._calculate_path_distance(
                hypothetical_agent_end,
                self.agent.shortest_path
            )
            progress = self._calculate_progress_score(path_distance_start, hypothetical_path_distance)
            
            # 计算cost分数
            cost = self._calculate_cost_score('MOVE')
            
            # 计算总分
            total = (
                self.safety_weight * safety +
                self.progress_weight * progress +
                self.cost_weight * cost
            )
            
            candidate_scores[action_id] = {
                'safety_score': safety,
                'progress_score': progress,
                'cost_score': cost,
                'total_score': total
            }
        
        stationary_safety = self._calculate_safety_score(
            agent_start_position,
            agent_start_position,
            npc_start_positions,
            npc_end_positions,
            dynamic_start_states,
            dynamic_end_states,
        )

        # Evaluate turns against moving hazards even though the agent remains
        # at the same XY location.
        for turn_id in ['L30', 'L60', 'L90', 'R30', 'R60', 'R90']:
            candidate_scores[turn_id] = {
                'safety_score': stationary_safety,
                'progress_score': 0.0,  # 没有前进
                'cost_score': self._calculate_cost_score('TURN'),
                'total_score': self.safety_weight * stationary_safety + self.cost_weight * self._calculate_cost_score('TURN')
            }
        
        # 评估等待 (WAIT_1/WAIT_2/WAIT_3)
        for wait_id in ['WAIT_1', 'WAIT_2', 'WAIT_3']:
            duration = int(wait_id.split('_')[1])
            candidate_scores[wait_id] = {
                'safety_score': stationary_safety,
                'progress_score': 0.0,
                'cost_score': self._calculate_cost_score('WAIT') * duration,
                'total_score': (
                    self.safety_weight * stationary_safety +
                    self.cost_weight * self._calculate_cost_score('WAIT') * duration
                )
            }
        
        return candidate_scores
    
    def _get_npc_positions(self) -> Dict[str, Vector]:
        """
        Return live positions for pedestrians and registered dynamic actors.
        
        Returns:
            npc_positions: 字典 {npc_name: Vector(x, y)}
        """
        # 获取所有对象
        all_objects = self.communicator.unrealcv.get_objects()
        
        # Discover spawned pedestrians/irregular NPCs and merge the explicit
        # benchmark actor IDs selected before RTAgent construction.
        npc_names = []
        for obj_name in all_objects:
            obj_name_lower = obj_name.lower()
            if obj_name_lower.startswith('rt_pedestrian_') or obj_name_lower.startswith('rt_scooter_'):
                npc_names.append(obj_name)
        available_names = set(all_objects)
        npc_names.extend(
            actor_id
            for actor_id in self.dynamic_obstacles
            if actor_id in available_names and actor_id not in npc_names
        )
        
        if not npc_names:
            # Do not leak coordinates from a previous frame when UE no longer
            # exposes a registered actor (for example after despawn/reset).
            self._latest_dynamic_states = {}
            return {}
        
        # 批量获取位置
        locations = self.communicator.unrealcv.get_location_batch(npc_names)
        
        # Build the existing XY interface while retaining XYZ for falling
        # hazards and activation verification.
        npc_positions = {}
        dynamic_states = {}
        for name, location in zip(npc_names, locations):
            npc_positions[name] = Vector(location[0], location[1])
            if name in self.dynamic_obstacles:
                xyz = (
                    float(location[0]),
                    float(location[1]),
                    float(location[2]) if len(location) >= 3 else 0.0,
                )
                dynamic_states[name] = xyz
                self._record_dynamic_actor_location(name, xyz)
        self._latest_dynamic_states = dynamic_states
        
        return npc_positions

    def _record_dynamic_actor_location(self, actor_id, location) -> None:
        record = self._dynamic_actor_telemetry.setdefault(actor_id, {
            'actor_id': actor_id,
            'kind': self.dynamic_obstacles.get(actor_id, {}).get(
                'kind', 'dynamic'
            ),
            'initial_location_cm': list(location),
            'baseline_live_location_cm': None,
            'last_location_cm': None,
            'max_displacement_cm': 0.0,
            'max_planar_displacement_cm': 0.0,
            'sample_count': 0,
            'observed_moving': False,
        })
        if record['baseline_live_location_cm'] is None:
            record['baseline_live_location_cm'] = [
                round(float(v), 4) for v in location
            ]
        initial = record['baseline_live_location_cm']
        displacement = float(np.linalg.norm(
            np.asarray(location, dtype=float)
            - np.asarray(initial, dtype=float)
        ))
        planar_displacement = float(np.linalg.norm(
            np.asarray(location[:2], dtype=float)
            - np.asarray(initial[:2], dtype=float)
        ))
        record['last_location_cm'] = [round(float(v), 4) for v in location]
        record['max_displacement_cm'] = round(max(
            float(record['max_displacement_cm']), displacement
        ), 4)
        record['max_planar_displacement_cm'] = round(max(
            float(record['max_planar_displacement_cm']), planar_displacement
        ), 4)
        record['sample_count'] += 1
        record['observed_moving'] = bool(
            record['max_displacement_cm'] > self.dynamic_movement_threshold_cm
        )

    def dynamic_actor_telemetry(self) -> Dict:
        actors = [
            dict(self._dynamic_actor_telemetry[actor_id])
            for actor_id in sorted(self._dynamic_actor_telemetry)
        ]
        observed = [item for item in actors if item['observed_moving']]
        sampled = [item for item in actors if item['sample_count'] > 0]
        return {
            'registered_count': len(actors),
            'sampled_live_count': len(sampled),
            'observed_moving_count': len(observed),
            'movement_threshold_cm': self.dynamic_movement_threshold_cm,
            'max_displacement_cm': max(
                (item['max_displacement_cm'] for item in actors),
                default=0.0,
            ),
            'actors': actors,
        }

    def _estimate_npc_velocities(self, npc_positions: Dict[str, Vector]) -> Dict[str, Vector]:
        """Estimate NPC velocities from the previous prompt-time location snapshot."""
        velocities = {}
        self._latest_dynamic_z_velocities = {}
        previous_positions = self.last_preview_npc_positions or {}
        previous_time = self.last_preview_sim_time
        current_time = getattr(self.agent, 'sim_time_elapsed', None)

        if previous_time is None or current_time is None:
            return velocities

        dt = current_time - previous_time
        if dt <= 0.05:
            return velocities

        for name, pos in npc_positions.items():
            prev = previous_positions.get(name)
            if prev is None:
                continue
            velocities[name] = Vector(
                (pos.x - prev.x) / dt,
                (pos.y - prev.y) / dt
            )
            if (
                name in self._latest_dynamic_states
                and name in self._last_context_dynamic_states
            ):
                self._latest_dynamic_z_velocities[name] = (
                    self._latest_dynamic_states[name][2]
                    - self._last_context_dynamic_states[name][2]
                ) / dt

        return velocities

    def _calculate_predicted_min_distance_details(
        self,
        agent_start: Vector,
        agent_end: Vector,
        npc_positions: Dict[str, Vector],
        npc_velocities: Dict[str, Vector],
        expected_latency: float,
        action_duration: float
    ) -> Tuple[float, List[Dict]]:
        """Calculate per-obstacle distances for the expected action time window."""
        obstacle_distances = []

        for npc_name, npc_position in npc_positions.items():
            velocity = npc_velocities.get(npc_name, Vector(0, 0))
            npc_start = Vector(
                npc_position.x + velocity.x * expected_latency,
                npc_position.y + velocity.y * expected_latency
            )
            npc_end = Vector(
                npc_position.x + velocity.x * (expected_latency + action_duration),
                npc_position.y + velocity.y * (expected_latency + action_duration)
            )
            distance = self._dynamic_relative_distance(
                npc_name,
                agent_start,
                agent_end,
                npc_start,
                npc_end,
                current_state=self._latest_dynamic_states.get(npc_name),
                vertical_velocity=self._latest_dynamic_z_velocities.get(
                    npc_name, 0.0
                ),
                expected_latency=expected_latency,
                action_duration=action_duration,
            )
            obstacle_distances.append({
                'kind': self.dynamic_obstacles.get(npc_name, {}).get(
                    'kind', 'dynamic'
                ),
                'id': npc_name,
                'distance_cm': distance
            })

        for obstacle in self.static_obstacles:
            distance = self._static_obstacle_distance(
                obstacle,
                agent_start,
                agent_end
            )
            obstacle_distances.append({
                'kind': obstacle.get('kind', 'static'),
                'id': obstacle.get('id', 'unknown'),
                'distance_cm': distance
            })

        if not obstacle_distances:
            return float('inf'), []

        obstacle_distances.sort(key=lambda item: item['distance_cm'])
        min_distance = obstacle_distances[0]['distance_cm']
        return min_distance, obstacle_distances[:3]

    def _dynamic_relative_distance(
        self,
        actor_id: str,
        agent_start: Vector,
        agent_end: Vector,
        actor_start: Vector,
        actor_end: Vector,
        current_state=None,
        vertical_velocity: float = 0.0,
        expected_latency: float = 0.0,
        action_duration: float = 0.0,
        actor_start_state=None,
        actor_end_state=None,
    ) -> float:
        """Relative trajectory distance, retaining Z for falling hazards."""
        kind = self.dynamic_obstacles.get(actor_id, {}).get('kind')
        if kind != 'falling_object':
            return self._calculate_relative_distance(
                agent_start, agent_end, actor_start, actor_end
            )

        if actor_start_state is not None and actor_end_state is not None:
            start_z = float(actor_start_state[2])
            end_z = float(actor_end_state[2])
        elif current_state is not None:
            start_z = float(current_state[2]) + (
                float(vertical_velocity) * float(expected_latency)
            )
            end_z = float(current_state[2]) + (
                float(vertical_velocity)
                * float(expected_latency + action_duration)
            )
        else:
            # Missing live Z must not turn an overhead falling object into an
            # immediate ground-plane collision. Its authored Z is the safest
            # available fallback.
            initial = self.dynamic_obstacles.get(actor_id, {}).get(
                'initial_location_cm', [actor_start.x, actor_start.y, 0.0]
            )
            start_z = end_z = float(initial[2])

        agent_start_3d = np.asarray(
            [agent_start.x, agent_start.y, 110.0], dtype=float
        )
        agent_end_3d = np.asarray(
            [agent_end.x, agent_end.y, 110.0], dtype=float
        )
        actor_start_3d = np.asarray(
            [actor_start.x, actor_start.y, start_z], dtype=float
        )
        actor_end_3d = np.asarray(
            [actor_end.x, actor_end.y, end_z], dtype=float
        )
        relative_start = agent_start_3d - actor_start_3d
        relative_end = agent_end_3d - actor_end_3d
        segment = relative_end - relative_start
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 1e-9:
            return float(np.linalg.norm(relative_start))
        projection = float(
            np.clip(-np.dot(relative_start, segment) / length_sq, 0.0, 1.0)
        )
        return float(np.linalg.norm(relative_start + projection * segment))

    def _assess_route_corridor(
        self,
        agent_start: Vector,
        agent_end: Vector
    ) -> Tuple[bool, Dict]:
        """
        Check whether a waypoint segment stays on or returns toward the planned
        sidewalk/crosswalk route.
        """
        route = self._get_route_polyline()
        if not route:
            return True, {}

        start_deviation = self._distance_to_route(agent_start, route)
        end_deviation = self._distance_to_route(agent_end, route)
        max_deviation = max(
            self._distance_to_route(point, route)
            for point in self._sample_segment(agent_start, agent_end, samples=5)
        )

        if start_deviation <= self.route_corridor_limit:
            feasible = max_deviation <= self.route_corridor_limit
        else:
            feasible = (
                end_deviation <= start_deviation - self.route_correction_margin and
                max_deviation <= start_deviation + self.route_corridor_limit
            )

        return feasible, {
            'kind': 'route_corridor',
            'id': 'planned_sidewalk_route',
            'distance_cm': 0.0,
            'start_deviation_cm': start_deviation,
            'end_deviation_cm': end_deviation,
            'max_deviation_cm': max_deviation,
        }

    def _sample_segment(self, start: Vector, end: Vector, samples: int = 5) -> List[Vector]:
        if samples <= 1:
            return [end]
        points = []
        for i in range(samples):
            t = i / (samples - 1)
            points.append(Vector(
                start.x + (end.x - start.x) * t,
                start.y + (end.y - start.y) * t
            ))
        return points

    def _distance_to_route(self, point: Vector, route: List[Vector]) -> float:
        if not route:
            return float('inf')
        if len(route) == 1:
            return point.distance(route[0])

        return min(
            self._point_to_segment_distance(point, route[i], route[i + 1])
            for i in range(len(route) - 1)
        )

    def _get_route_polyline(self) -> List[Vector]:
        route = []
        start_pos = getattr(self.agent, 'start_pos', None)
        if start_pos is not None:
            route.append(start_pos)
        route.extend(self.agent.shortest_path or [])
        return route

    def _remaining_route_polyline(self) -> List[Vector]:
        route = [self.agent.position]
        route.extend(self.agent.shortest_path or [])
        if len(route) == 1 and getattr(self.agent, 'destination', None) is not None:
            route.append(self.agent.destination)
        return route

    def _polyline_length(self, route: List[Vector]) -> float:
        if len(route) < 2:
            return 0.0
        return sum(route[i].distance(route[i + 1]) for i in range(len(route) - 1))

    def _advance_along_polyline(
        self,
        current: Vector,
        route: List[Vector],
        route_index: int,
        distance: float
    ) -> Tuple[Vector, int]:
        remaining = distance
        point = current
        index = min(max(route_index, 0), max(len(route) - 2, 0))

        while index < len(route) - 1:
            segment_end = route[index + 1]
            segment_distance = point.distance(segment_end)
            if segment_distance <= 1e-6:
                point = segment_end
                index += 1
                continue
            if remaining <= segment_distance:
                ratio = remaining / segment_distance
                return Vector(
                    point.x + (segment_end.x - point.x) * ratio,
                    point.y + (segment_end.y - point.y) * ratio
                ), index
            remaining -= segment_distance
            point = segment_end
            index += 1

        return route[-1], max(len(route) - 2, 0)

    def _find_safe_wait_for_trajectory(
        self,
        step_num: int,
        current: Vector,
        npc_positions: Dict[str, Vector],
        npc_velocities: Dict[str, Vector],
        prediction_start_time: float,
        cumulative_time: float
    ) -> Dict | None:
        best_record = None
        best_clearance = -1.0
        for wait_duration in [1.0, 2.0, 3.0]:
            min_distance, closest_obstacles = self._calculate_predicted_min_distance_details(
                current,
                current,
                npc_positions,
                npc_velocities,
                prediction_start_time,
                wait_duration
            )
            safety = self._score_from_min_distance(min_distance)
            if self._is_safe_clearance(min_distance, safety) and min_distance > best_clearance:
                best_clearance = min_distance
                best_record = self._trajectory_step_record(
                    step_num,
                    WAIT,
                    str(int(wait_duration)),
                    current,
                    current,
                    wait_duration,
                    cumulative_time + wait_duration,
                    min_distance,
                    closest_obstacles
                )
        return best_record

    def _trajectory_step_record(
        self,
        step_num: int,
        action_type: str,
        action_param: str | None,
        start: Vector,
        end: Vector,
        duration: float,
        cumulative_time: float,
        min_distance: float,
        closest_obstacles: List[Dict]
    ) -> Dict:
        closest = closest_obstacles[0] if closest_obstacles else None
        return {
            'step': step_num,
            'action_type': action_type,
            'action_param': action_param,
            'start': self._point_dict(start),
            'end': self._point_dict(end),
            'duration_s': round(duration, 2),
            'cumulative_time_s': round(cumulative_time, 2),
            'min_clearance_cm': self._json_float(min_distance),
            'closest_obstacle': self._obstacle_dict(closest),
        }

    def _point_dict(self, point: Vector) -> Dict:
        return {'x': round(point.x, 4), 'y': round(point.y, 4)}

    def _obstacle_dict(self, obstacle: Dict | None) -> Dict | None:
        if not obstacle:
            return None
        return {
            'kind': obstacle.get('kind'),
            'id': obstacle.get('id'),
            'distance_cm': self._json_float(obstacle.get('distance_cm', float('inf'))),
        }

    def _json_float(self, value: float) -> float | str:
        if value == float('inf'):
            return 'inf'
        return round(float(value), 4)

    def _score_from_min_distance(self, min_distance: float) -> float:
        """Map a minimum clearance distance to the evaluator's barrier-style score."""
        if min_distance == float('inf'):
            return 1.0
        if min_distance < self.safety_thresholds['danger']:
            return 0.0
        if min_distance < self.safety_thresholds['unsafe']:
            return 0.5 * (min_distance - self.safety_thresholds['danger']) / \
                (self.safety_thresholds['unsafe'] - self.safety_thresholds['danger'])
        if min_distance < self.safety_thresholds['safe']:
            return 0.5 + 0.4 * (min_distance - self.safety_thresholds['unsafe']) / \
                (self.safety_thresholds['safe'] - self.safety_thresholds['unsafe'])
        return 1.0

    def _is_safe_clearance(self, min_distance: float, safety_score: float) -> bool:
        """Treat actions above the unsafe distance as barrier-feasible."""
        return (
            min_distance == float('inf') or
            (min_distance >= self.safety_thresholds['unsafe'] and safety_score >= 0.5)
        )

    def _format_distance(self, distance: float) -> str:
        if distance == 'inf':
            return "inf"
        if distance == float('inf'):
            return "inf"
        return f"{distance:.0f}cm"

    def _ordered_action_ids(self) -> List[str]:
        return [
            '1', '2', '3', '4', '5', '6', '7',
            'L30', 'L60', 'L90', 'R30', 'R60', 'R90',
            'WAIT_1', 'WAIT_2', 'WAIT_3'
        ]
    
    def _calculate_path_distance(self, current_pos: Vector, path: List[Vector]) -> float:
        """
        计算沿着路径到终点的距离
        
        Args:
            current_pos: 当前位置
            path: 路径点列表
            
        Returns:
            distance: 沿路径的总距离
        """
        if not path:
            return 0.0
        
        # 距离 = 当前位置到第一个路径点 + 路径点之间的距离
        total_distance = current_pos.distance(path[0])
        
        for i in range(len(path) - 1):
            total_distance += path[i].distance(path[i + 1])
        
        return total_distance
    
    def _calculate_safety_score(
        self,
        agent_start: Vector,
        agent_end: Vector,
        npc_start_positions: Dict[str, Vector],
        npc_end_positions: Dict[str, Vector],
        dynamic_start_states: Dict | None = None,
        dynamic_end_states: Dict | None = None,
    ) -> float:
        """
        计算safety分数：基于agent轨迹与NPC轨迹、静态障碍物的最短距离
        
        Args:
            agent_start: agent起始位置
            agent_end: agent结束位置
            npc_start_positions: NPC起始位置字典
            npc_end_positions: NPC结束位置字典
            
        Returns:
            safety_score: 安全分数 (0-1之间，越高越安全)
        """
        min_distance = float('inf')
        
        # 1. 计算与动态NPC的最短距离
        if npc_start_positions and npc_end_positions:
            for npc_name in npc_start_positions:
                if npc_name not in npc_end_positions:
                    continue
                
                npc_start = npc_start_positions[npc_name]
                npc_end = npc_end_positions[npc_name]
                
                # 使用相对位置计算距离（消除时间差问题）
                distance = self._dynamic_relative_distance(
                    npc_name,
                    agent_start,
                    agent_end,
                    npc_start,
                    npc_end,
                    actor_start_state=(dynamic_start_states or {}).get(
                        npc_name
                    ),
                    actor_end_state=(dynamic_end_states or {}).get(npc_name),
                )
                
                min_distance = min(min_distance, distance)
        
        # 2. 计算与静态障碍物的最短距离
        if self.static_obstacles:
            for obstacle in self.static_obstacles:
                distance = self._static_obstacle_distance(obstacle, agent_start, agent_end)
                min_distance = min(min_distance, distance)
        
        # 记录最小距离用于日志
        self._min_distance = min_distance if min_distance != float('inf') else None
        
        return self._score_from_min_distance(min_distance)
    
    def _calculate_relative_distance(
        self,
        agent_start: Vector,
        agent_end: Vector,
        npc_start: Vector,
        npc_end: Vector
    ) -> float:
        """
        使用相对位置计算agent轨迹与NPC轨迹的最短距离
        通过将坐标转换到"NPC静止"的相对坐标系下，消除时间差问题
        
        Args:
            agent_start: agent起始位置
            agent_end: agent结束位置
            npc_start: NPC起始位置
            npc_end: NPC结束位置
            
        Returns:
            distance: 最短距离
        """
        # 1. 转换到NPC静止的相对坐标系
        # 转换为numpy数组进行计算
        agent_start_np = np.array([agent_start.x, agent_start.y])
        agent_end_np = np.array([agent_end.x, agent_end.y])
        npc_start_np = np.array([npc_start.x, npc_start.y])
        npc_end_np = np.array([npc_end.x, npc_end.y])
        
        rel_start = agent_start_np - npc_start_np
        rel_end = agent_end_np - npc_end_np
        
        # 2. 问题简化为：原点(0,0)到线段(rel_start -> rel_end)的最短距离
        
        # 线段向量
        line_vec = rel_end - rel_start
        
        # 原点到起点的向量（因为原点是0,0，所以就是-rel_start）
        point_vec = -rel_start
        
        # 线段长度的平方
        line_len_sq = np.dot(line_vec, line_vec)
        
        if line_len_sq == 0:
            # 起点终点重合，直接算距离
            return np.linalg.norm(rel_start)
        
        # 3. 计算投影比例t（t必须在0到1之间，才算在线段上）
        t = np.dot(point_vec, line_vec) / line_len_sq
        t = np.clip(t, 0.0, 1.0)  # 限制t在[0, 1]范围内
        
        # 4. 找到线段上离原点最近的点
        closest_point = rel_start + line_vec * t
        
        # 5. 计算距离
        distance = np.linalg.norm(closest_point)
        
        return distance

    def _static_obstacle_distance(
        self,
        obstacle: Dict,
        segment_start: Vector,
        segment_end: Vector
    ) -> float:
        """Distance from an agent path segment to a static point obstacle or AABB."""
        bounds = obstacle.get('bounds')
        if bounds:
            return self._segment_to_aabb_distance(segment_start, segment_end, bounds)

        obstacle_pos = Vector(obstacle['x'], obstacle['y'])
        return self._point_to_segment_distance(obstacle_pos, segment_start, segment_end)

    def _segment_to_aabb_distance(
        self,
        segment_start: Vector,
        segment_end: Vector,
        bounds: Dict
    ) -> float:
        """Return 0 when a path segment intersects an axis-aligned building box."""
        min_x = min(bounds['min_x'], bounds['max_x'])
        max_x = max(bounds['min_x'], bounds['max_x'])
        min_y = min(bounds['min_y'], bounds['max_y'])
        max_y = max(bounds['min_y'], bounds['max_y'])

        if self._segment_intersects_aabb(segment_start, segment_end, min_x, min_y, max_x, max_y):
            return 0.0

        corners = [
            Vector(min_x, min_y),
            Vector(max_x, min_y),
            Vector(max_x, max_y),
            Vector(min_x, max_y),
        ]
        edges = list(zip(corners, corners[1:] + corners[:1]))
        distances = [
            self._point_to_segment_distance(corner, segment_start, segment_end)
            for corner in corners
        ]
        distances.extend(
            self._segment_to_segment_distance(segment_start, segment_end, edge_start, edge_end)
            for edge_start, edge_end in edges
        )
        return min(distances)

    def _segment_intersects_aabb(
        self,
        segment_start: Vector,
        segment_end: Vector,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float
    ) -> bool:
        """Liang-Barsky segment/AABB intersection."""
        x0, y0 = segment_start.x, segment_start.y
        x1, y1 = segment_end.x, segment_end.y

        if min_x <= x0 <= max_x and min_y <= y0 <= max_y:
            return True
        if min_x <= x1 <= max_x and min_y <= y1 <= max_y:
            return True

        dx = x1 - x0
        dy = y1 - y0
        p = [-dx, dx, -dy, dy]
        q = [x0 - min_x, max_x - x0, y0 - min_y, max_y - y0]
        u1, u2 = 0.0, 1.0

        for pi, qi in zip(p, q):
            if pi == 0:
                if qi < 0:
                    return False
                continue
            ratio = qi / pi
            if pi < 0:
                u1 = max(u1, ratio)
            else:
                u2 = min(u2, ratio)
            if u1 > u2:
                return False

        return True

    def _segment_to_segment_distance(
        self,
        a_start: Vector,
        a_end: Vector,
        b_start: Vector,
        b_end: Vector
    ) -> float:
        """Shortest distance between two 2D line segments."""
        if self._segments_intersect(a_start, a_end, b_start, b_end):
            return 0.0
        return min(
            self._point_to_segment_distance(a_start, b_start, b_end),
            self._point_to_segment_distance(a_end, b_start, b_end),
            self._point_to_segment_distance(b_start, a_start, a_end),
            self._point_to_segment_distance(b_end, a_start, a_end),
        )

    def _segments_intersect(
        self,
        a_start: Vector,
        a_end: Vector,
        b_start: Vector,
        b_end: Vector
    ) -> bool:
        def orient(p: Vector, q: Vector, r: Vector) -> float:
            return (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x)

        def on_segment(p: Vector, q: Vector, r: Vector) -> bool:
            return (
                min(p.x, r.x) <= q.x <= max(p.x, r.x) and
                min(p.y, r.y) <= q.y <= max(p.y, r.y)
            )

        o1 = orient(a_start, a_end, b_start)
        o2 = orient(a_start, a_end, b_end)
        o3 = orient(b_start, b_end, a_start)
        o4 = orient(b_start, b_end, a_end)
        eps = 1e-9

        if o1 * o2 < 0 and o3 * o4 < 0:
            return True
        if abs(o1) <= eps and on_segment(a_start, b_start, a_end):
            return True
        if abs(o2) <= eps and on_segment(a_start, b_end, a_end):
            return True
        if abs(o3) <= eps and on_segment(b_start, a_start, b_end):
            return True
        if abs(o4) <= eps and on_segment(b_start, a_end, b_end):
            return True
        return False
    
    def _point_to_segment_distance(
        self,
        point: Vector,
        segment_start: Vector,
        segment_end: Vector
    ) -> float:
        """
        计算点到线段的最短距离
        
        Args:
            point: 点的位置
            segment_start: 线段起点
            segment_end: 线段终点
            
        Returns:
            distance: 最短距离
        """
        # 转换为numpy数组
        p = np.array([point.x, point.y])
        a = np.array([segment_start.x, segment_start.y])
        b = np.array([segment_end.x, segment_end.y])
        
        # 线段向量
        ab = b - a
        # 点到起点的向量
        ap = p - a
        
        # 线段长度的平方
        ab_len_sq = np.dot(ab, ab)
        
        if ab_len_sq == 0:
            # 起点终点重合，直接算距离
            return np.linalg.norm(ap)
        
        # 计算投影比例t
        t = np.dot(ap, ab) / ab_len_sq
        t = np.clip(t, 0.0, 1.0)  # 限制在线段上
        
        # 线段上最近的点
        closest_point = a + ab * t
        
        # 计算距离
        distance = np.linalg.norm(p - closest_point)
        
        return distance
    
    def _calculate_progress_score(
        self,
        path_distance_before: float,
        path_distance_after: float
    ) -> float:
        """
        计算progress分数：离终点近了多少
        只有前进（距离减少）才有正分，其他为0分
        
        Args:
            path_distance_before: 动作前的路径距离
            path_distance_after: 动作后的路径距离
            
        Returns:
            progress_score: 进度分数（归一化到0-1之间）
        """
        # 计算距离变化
        distance_change = path_distance_before - path_distance_after
        
        # 只有前进才有正分
        if distance_change <= 0:
            return 0.0
        
        # 归一化：假设一步最多前进300cm（agent.speed * 2）
        # 根据实际前进距离给分
        max_progress = 300.0  # 可以根据agent.speed调整
        score = min(1.0, distance_change / max_progress)
        
        return score
    
    def _calculate_cost_score(self, action_type: str) -> float:
        """
        计算cost分数：根据动作类型给予不同惩罚
        
        Args:
            action_type: 动作类型 ('MOVE', 'TURN', 'WAIT')
            
        Returns:
            cost_score: 成本分数（负值表示惩罚）
        """
        return self.cost_penalties.get(action_type, 0.0)
    
    def get_summary_statistics(
        self,
        attempted_decisions: int | None = None,
        invalid_decisions: int = 0,
    ) -> Dict:
        """
        获取评测统计摘要
        
        Returns:
            summary: 统计信息字典
        """
        attempted = (
            max(0, int(attempted_decisions))
            if attempted_decisions is not None
            else len(self.step_records)
        )
        evaluated = len(self.step_records)
        invalid = max(0, int(invalid_decisions or 0))
        coverage = {
            'attempted_decisions': attempted,
            'evaluated_decisions': evaluated,
            'unevaluated_decisions': max(0, attempted - evaluated),
            'invalid_decisions': invalid,
            'evaluation_coverage_rate': evaluated / attempted if attempted else 0.0,
            'optimal_choice_rate_all_decisions': 0.0,
        }

        if not self.step_records:
            return {
                'total_steps': 0,
                'evaluation_coverage': coverage,
            }
        
        safety_scores = [r['safety_score'] for r in self.step_records]
        progress_scores = [r['progress_score'] for r in self.step_records]
        cost_scores = [r['cost_score'] for r in self.step_records]
        total_scores = [r['total_score'] for r in self.step_records]
        
        # 计算选择质量统计
        is_optimal_list = [r.get('is_optimal', False) for r in self.step_records]
        score_gaps = [r.get('score_gap', 0) for r in self.step_records]
        
        optimal_count = sum(is_optimal_list)
        optimal_rate = optimal_count / len(self.step_records) if self.step_records else 0
        coverage['optimal_choice_rate_all_decisions'] = (
            optimal_count / attempted if attempted else 0.0
        )
        
        summary = {
            'total_steps': len(self.step_records),
            'safety': {
                'mean': np.mean(safety_scores),
                'std': np.std(safety_scores),
                'min': np.min(safety_scores),
                'max': np.max(safety_scores)
            },
            'progress': {
                'mean': np.mean(progress_scores),
                'std': np.std(progress_scores),
                'min': np.min(progress_scores),
                'max': np.max(progress_scores),
                'total': sum(progress_scores)
            },
            'cost': {
                'mean': np.mean(cost_scores),
                'std': np.std(cost_scores),
                'min': np.min(cost_scores),
                'max': np.max(cost_scores),
                'total': sum(cost_scores)
            },
            'total': {
                'mean': np.mean(total_scores),
                'std': np.std(total_scores),
                'min': np.min(total_scores),
                'max': np.max(total_scores)
            },
            'action_distribution': self._get_action_distribution(),
            'decision_quality': {
                'optimal_choice_count': optimal_count,
                'optimal_choice_rate': optimal_rate,
                'average_score_gap': np.mean(score_gaps),
                'max_score_gap': np.max(score_gaps) if score_gaps else 0,
                'total_score_gap': sum(score_gaps)
            },
            'evaluation_coverage': coverage,
        }
        
        return summary
    
    def _get_action_distribution(self) -> Dict[str, int]:
        """获取动作类型分布"""
        distribution = {}
        for record in self.step_records:
            action_type = record['action_type']
            distribution[action_type] = distribution.get(action_type, 0) + 1
        return distribution
