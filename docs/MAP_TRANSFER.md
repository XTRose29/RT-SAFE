# Extending RT-SAFE to another map

## Current outcome

An isolated NYC pilot runs in the native Madison Square Park Unreal project.
It supports the RT-SAFE action interface, physical agent movement, moving
pedestrians, robot dogs, vehicles, movable objects, environmental hazards,
falling objects, and annotated traffic rules. A combined sidewalk/crossing
route exercises these components together. This is an implementation and validation pilot;
it does not extend the published five-map result tables.

The original levels (`RT10`, `RT12`, `RT15`, `RT18`, `RT20`) and their task files
remain unchanged. The new adapter lives in `benchmark/map_transfer/`.
See its [setup guide](../benchmark/map_transfer/README.md) for runnable commands.
The [validation report](NYC_TRANSFER_RESULTS.md) records the fresh rebuild,
26 physical checks, action calibration, and bounded visual-agent trials.

## Why changing the map name is insufficient

The available old environment is a cooked UnrealCV runtime. Its RT-SAFE
Blueprints cannot be loaded into the available NYC editor. NYC uses a
project-matched Unreal 5.8 preview build and SPEAR v1.0.0. Transport, agent
control, collision geometry, actor spawning, traffic annotations, and reset
behavior therefore need an explicit adapter.

Two routes are possible:

- **Preserve the exact old runtime:** obtain uncooked RT-SAFE Blueprint/C++
  sources and compatible plugins, migrate them into an engine-compatible NYC
  project, then audit behavior against recorded old-map episodes. This is the
  preferred route for a directly comparable expanded benchmark, but the
  required uncooked sources are not present in the supplied runtime.
- **Implement a native adapter:** preserve the public action and safety
  concepts while using the available NYC engine interfaces. This is the route
  implemented here. Differences are recorded in every run's provenance.

## Transfer plan and acceptance checks

| Stage | Implementation | Acceptance evidence |
| --- | --- | --- |
| Preserve the baseline | Separate adapter and owned project; no edits to five-map tasks or results | Git diff and source-package checksums |
| Prepare geometry | Duplicate selected static meshes into `/Game/RTSafeNYC/`; use triangle collision | Ground/capsule route audit and actual traversal |
| Place tasks | Explicit centimeter coordinates, start/goal, entity paths, road and crosswalk polygons | Native camera inspection and physical route checks |
| Control the agent | Seven moves, six turns, three waits; native blocking CharacterMovement | Clear route, box blocking, native wall blocking |
| Move other actors | Pedestrian CharacterMovement, swept robot/vehicle bodies, rigid-body objects | Real-time contact, static freeze, movable-object displacement |
| Connect traffic events | Authored conflict vehicle activates after a violation in its linked zone | Red crossing produces contact; a legal crossing leaves the vehicle inactive |
| Detect safety events | Native blocking hits, continuous hazard overlaps, traffic-zone entry | Positive and negative cases; sustained contact and recontact |
| Preserve timing intent | Async real-time simulation; pause only during static inference | Simulation/wall clock measurements and actor displacement |
| Reset repeatedly | Clean up owned actors and release PIE callbacks between episodes | Multiple successive episodes in one editor |
| Exercise a visual policy | Bounded inexpensive API and Codex CLI policies | Saved RGB, actions, timestamps, native events, model usage |
| Qualify an expanded benchmark | Audit remaining protocol differences and author a representative task set | Separate future release gate; not established by this pilot |

## What had to change in NYC

### Collision geometry

Some district-sized imported static meshes had collision that enclosed empty
sidewalks. Four park detail meshes similarly blocked a visually open path.
The adapter creates owned copies and replaces their collision with the actual
triangle surface. It does not remove collision from the environment.

Preparation hashes matched original mesh packages and the original map
before and after saving the derivative. The original assets must remain
unchanged. Triangle collision is only used for selected static geometry;
moving physics bodies retain simple colliders.

A sampled route audit is useful for locating problems but is not a proof of
navigability. In particular, capsule overlap samples can flag a curb that
CharacterMovement successfully steps over. Confirm routes through execution.
Object placement also samples the complete box footprint, so a prop placed
on sloping pavement does not start inside the ground. Initial capsule and
fixture overlaps are rejected before an episode begins.

### Runtime and reset behavior

Film-rendering fixed-step settings initially advanced the simulation too
slowly relative to wall time. The pilot explicitly uses asynchronous stepping,
disables benchmarking and fixed-delta overrides, and measures clock behavior.

The SPEAR connection retains a world context. Saving a derivative is safe;
switching maps inside a live RPC invalidates that context. Restart the owned
editor to load the saved map. Python wrappers for PIE delegates also must be
released before the PIE world is destroyed. The adapter unbinds callbacks,
collects wrappers, waits for PIE to end, and clears owned scene fixtures.

A visual-policy trial also exposed a control bug after pedestrian contact:
repeated hit callbacks canceled even movement away from the pedestrian. The
controller now uses the native impact normal to distinguish movement into the
contact from sideways or backward escape. Two dedicated recovery cases verify
that the agent can leave contact while preserving the original collision event.

### Map-specific traffic and hazards

The original procedural road coordinates do not describe NYC. Each crossing
uses authored road/crosswalk polygons aligned with the native street markings,
a visible pedestrian signal, and an explicit signal schedule. Ambient
MassTraffic and crowd spawners are disabled for the pilot fixtures.

Hazard volumes use native overlap queries. Trip events impose six simulated
seconds of recovery; oil slows the next move; water records an event without
changing movement. These settings are explicit, and do not by themselves
establish equivalence to every historical campaign.

## Before treating NYC as a sixth benchmark map

- Obtain the original uncooked runtime or agree on a versioned replacement
  protocol. Reconcile collision-entry counting with the old callback counts.
- Match observation history and prompts. The pilot policies currently use a
  single current RGB image, goal-relative information, and recent textual
  action feedback; the paper protocol also includes previous-action frames.
- Author routes across multiple districts with sequential navigation goals,
  rather than interpreting the small fixture set as representative coverage.
- Implement and validate the original easy/medium/hard actor activation
  protocol, with controlled paired seeds and documented actor placement.
- Validate vehicle, robot, and pedestrian dynamics against the intended
  benchmark. Swept bodies and visual gait animation are explicit pilot
  approximations, not a full traffic or legged-robot physics model.
- Certify navigable shortest paths before reporting SPL. The pilot reports
  `null` for SPL.
- Run repeated paired static/real-time trials, report failures and timing
  variation, and retain exact engine, SDK, adapter, manifest, and model versions.
- Resolve redistribution rights and provide a versioned runtime package or
  reproducible licensed-asset setup. This repository does not contain the
  proprietary city assets, native binaries, or SDK copy.

The website's NYC environment explorer is a separate presentation component.
Rendered or reconstructed demonstration clips are not evidence that the
native benchmark passed these checks.
