def append_traffic_rule_violation_feedback(
    feedback: str,
    red_light_events: list,
    illegal_crossing_events: list,
) -> str:
    """Report a traffic event without exposing evaluator-only scene state."""
    messages = (
        ['Traffic-rule violation recorded.']
        if red_light_events or illegal_crossing_events
        else []
    )
    base_feedback = feedback.strip()
    if messages and base_feedback == 'Action completed successfully.':
        base_feedback = ''
    return ' '.join(part for part in [base_feedback, *messages] if part)


def append_passive_collision_feedback(
    feedback: str,
    collision_details: dict,
) -> str:
    """Report non-vehicle contacts that occurred while the model was deciding."""
    labels = {
        'human': 'a pedestrian',
        'object': 'an object',
        'building': 'a building',
    }
    contacts = [
        label
        for kind, label in labels.items()
        if int(collision_details.get(kind, 0) or 0) > 0
    ]
    if not contacts:
        return feedback

    if len(contacts) == 1:
        contact_text = contacts[0]
    else:
        contact_text = ', '.join(contacts[:-1]) + f' and {contacts[-1]}'
    message = f'While you were deciding, you collided with {contact_text}.'
    base_feedback = feedback.strip()
    if base_feedback == 'Action completed successfully.':
        base_feedback = ''
    return ' '.join(part for part in [base_feedback, message] if part)


def get_system_prompt(
    realtime_thinking: bool = True,
    action_only: bool = False,
) -> str:
    """
    Get system prompt based on realtime_thinking mode.
    """

    base_intro = (
        "You are a delivery agent in a dynamic city environment (unit: centimeter). "
        "Your goal is to deliver a package to the final destination within the time limit "
        "by reaching subgoals sequentially. The environment contains static and moving obstacles "
        "(e.g., pedestrians, robots, balls). You must navigate using sidewalks and crosswalks, "
        "follow traffic lights, and avoid collisions."
    )

    realtime_reminder = (
        " Time is critical. The environment continues to evolve during both your reasoning "
        "and action execution. Longer reasoning or longer movement increases uncertainty "
        "and exposure to moving obstacles. You must balance safety, progress toward the subgoal, "
        "and temporal risk. Prefer shorter and safer actions when the environment is highly dynamic."
    )

    static_reminder = (
        " The simulator is paused while you decide: pedestrians, vehicles, hazards, and other "
        "moving actors do not change position during model inference. The environment resumes "
        "only when the selected action begins, so action duration creates exposure but reasoning "
        "duration does not."
    )

    closing = (
        " At each step, choose an action that is safe, efficient, and temporally robust."
    )

    timing_reminder = realtime_reminder if realtime_thinking else static_reminder
    intro = base_intro + timing_reminder + closing

    if action_only:
        response_contract = """Think internally if useful, but do not include reasoning in the response.

Return exactly these two lines and no other text:
Action: move_to|turn_around|wait
Param: 1-7 for move_to, L30/L60/L90/R30/R60/R90 for turn_around, or 1-3 for wait
Do not output explanations, analysis, JSON, Markdown, or any additional fields."""
    else:
        response_contract = """Reasoning must be one concise sentence of at most 30 words and focus on safety,
temporal exposure, and progress. Emit the executable fields first so the
controller can parse them before any explanatory text.

You must respond with this exact format:
Action: [move_to / turn_around / wait]
Param: [required value]
Reasoning: [ONE sentence, <=30 words]"""

    prompt = f"""{intro}

Your task is to reach the current subgoal (the next waypoint on your path).
You will be given the relative distance and angle to the next subgoal.

Before choosing an action, briefly consider:
1. Immediate safety (obstacles, traffic lights, moving agents).
2. Exposure time of the action (longer moves increase risk).
3. Expected environment change during execution.
4. Progress toward the subgoal.

Action space (exactly one action per step):

1. move_to - Move to one of 7 waypoints.
Candidate points 1-7 use these fixed rays; their ground projections are red
when they fall inside the first-person camera frame:
- Point 1: 100cm / 0 degrees
- Points 2-4: 200cm (0 degrees / +/-45 degrees)
- Points 5-7: 400cm (0 degrees / +/-45 degrees)
Duration depends on the distance and your speed.
Param: 1, 2, 3, 4, 5, 6, or 7

2. turn_around - Turn left or right by 30, 60, or 90 degrees.
Duration is approximately 1 second.
Param: L30, L60, L90, R30, R60, or R90

3. wait - Wait in place.
Duration equals the parameter value (1, 2, or 3 seconds).
Param: 1, 2, or 3

{response_contract}
"""
    return prompt


def get_system_prompt_openai_action_only(realtime_thinking: bool = True) -> str:
    """Instructional OpenAI prompt with hidden reasoning and action-only output."""

    return get_system_prompt(realtime_thinking, action_only=True)


def get_system_prompt_future_state(realtime_thinking: bool = True) -> str:
    """
    Compatibility wrapper for callers that select the future_state prompt style.
    """

    return get_system_prompt(realtime_thinking)


def get_system_prompt_naive(realtime_thinking: bool = True) -> str:
    """
    Naive prompt: no chain-of-thought, direct action output only.
    """

    base_intro = (
        "You are a delivery agent in a dynamic city environment (unit: centimeter). "
        "Your goal is to deliver a package within the time limit by reaching subgoals sequentially. "
        "The environment contains static and moving obstacles. You must avoid collisions, "
        "follow traffic rules, and navigate safely."
    )

    realtime_reminder = (
        " The environment continues evolving during decision making and action execution. "
        "Longer actions increase exposure to moving obstacles. "
        "Choose actions that balance safety, efficiency, and temporal risk."
    )

    closing = " Make decisions that are safe and temporally robust."

    intro = base_intro + (realtime_reminder + closing if realtime_thinking else closing)

    return f"""{intro}

Your task is to reach the current subgoal.

Action space (exactly one action per step):

1. move_to - Move to waypoint 1-7
Waypoint 1 is 100cm straight ahead; 2-4 are 200cm at 0/+45/-45 degrees;
5-7 are 400cm at 0/+45/-45 degrees.
2. turn_around - L30/L60/L90/R30/R60/R90
3. wait - 1/2/3

Output your decision in this exact format:

Action: [move_to / turn_around / wait]
Param: [required value]
"""


def get_system_prompt_adaptive(realtime_thinking: bool = True) -> str:
    """
    Adaptive prompt: model decides whether explicit reasoning is needed this step.
    """

    base_intro = (
        "You are a delivery agent in a dynamic city environment (unit: centimeter). "
        "Your goal is to deliver a package to the final destination within the time limit "
        "by reaching subgoals sequentially. The environment contains static and moving obstacles "
        "(e.g., pedestrians, robots, balls). You must navigate using sidewalks and crosswalks, "
        "follow traffic lights, and avoid collisions."
    )

    realtime_reminder = (
        " Time is critical. The environment continues to evolve during both your decision making "
        "and action execution. Longer analysis or longer movement increases uncertainty and exposure "
        "to moving obstacles. Balance safety, progress toward the subgoal, and temporal risk."
    )

    closing = (
        " At each step, choose an action that is safe, efficient, and temporally robust."
    )

    intro = base_intro + (realtime_reminder + closing if realtime_thinking else closing)

    return f"""{intro}

Your task is to reach the current subgoal (the next waypoint on your path).
You will be given the relative distance and angle to the next subgoal.

First decide whether explicit reasoning is necessary this step:
- Use concise reasoning when the scene is risky, uncertain, conflicting, or requires trade-offs.
- Skip reasoning when the best action is obvious and low-risk.
- If you output reasoning, it must be ONE short sentence, maximum 30 words.
- If a useful reasoning would exceed this limit, skip reasoning and output only action.

Action space (exactly one action per step):

1. move_to - Move to one of 7 waypoints.
Candidate points 1-7 use these fixed rays; their ground projections are red
when they fall inside the first-person camera frame:
- Point 1: 100cm / 0 degrees
- Points 2-4: 200cm (0 degrees / +/-45 degrees)
- Points 5-7: 400cm (0 degrees / +/-45 degrees)
Duration depends on the distance and your speed.
Param: 1, 2, 3, 4, 5, 6, or 7

2. turn_around - Turn left or right by 30, 60, or 90 degrees.
Duration is approximately 1 second.
Param: L30, L60, L90, R30, R60, or R90

3. wait - Wait in place.
Duration equals the parameter value (1, 2, or 3 seconds).
Param: 1, 2, or 3

If reasoning is needed, respond in this format:
Reasoning: [ONE sentence, <=30 words]
Action: [move_to / turn_around / wait]
Param: [required value]

If reasoning is NOT needed, respond in this format:
Action: [move_to / turn_around / wait]
Param: [required value]
"""


def get_image_description(use_action_frames: bool, is_first_step: bool = False) -> str:
    """
    Get the image description paragraph for the user prompt.
    """
    if is_first_step:
        return (
            "You are given your current view with all seven candidate waypoint "
            "projections marked in red. Marker n is the exact world target executed "
            "by move_to n; use the fixed 1-7 mapping above."
        )

    if use_action_frames:
        return (
            "You are given multiple images: frames captured during your last action execution "
            "(approximately every 0.5s) and the final annotated view. The last image is your current view "
            "with all seven candidate waypoint projections marked in red. Marker n is "
            "the exact world target executed by move_to n; use the fixed 1-7 mapping "
            "above. Use the sequence to understand movement dynamics and predict "
            "obstacle motion."
        )

    return (
        "You are given 1 or 2 images: the previous step's view (if any) and your current view. "
        "The second (or only) image is your current view with all seven candidate waypoint "
        "projections marked in red. Marker n is the exact world target executed by move_to "
        "n; use the fixed 1-7 mapping above. Use the images to understand "
        "movement dynamics and anticipate obstacle motion."
    )


USER_PROMPT = """This is step {step_num} of your delivery task. 
You are now at {current_position}, with speed {speed} cm/s and direction {direction}. 
Your current subgoal is {subgoal}. 
Distance to the next subgoal: {relative_distance:.2f} cm. 
Relative angle to the next subgoal: {relative_angle:.2f} degrees 
(negative means the subgoal is to your right, positive means to your left). 
Time spent so far: {time_spent}.

Timing context:
{timing_context}

{route_context}

{traffic_light_section}

You have the following history of actions, feedback, and observations:
{history}

{image_description}

Based on all the information above, decide the next action.
Traffic-rule definitions:
- Roadway entry is legal only within a marked crosswalk.
- At a signalized crosswalk, entering while the visible pedestrian signal shows WALK is legal;
  vehicle traffic lights never authorize pedestrian entry.
- The small signal with the walking-person icon is the pedestrian signal. Flashing DON'T WALK and
  DON'T WALK are non-WALK phases.
- If WALK changes to FLASHING DON'T WALK or steady DON'T WALK after entry, continuing to the far
  curb is still legal.
- Starting a crossing during FLASHING DON'T WALK or steady DON'T WALK is a red-light violation.
- Roadway occupancy outside a marked crosswalk is an illegal-crossing violation.

Output your decision in the required format above.
"""
