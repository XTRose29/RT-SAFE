# Unreal runtime and assets

Live evaluation requires the RT-SAFE-compatible SimWorld Unreal Engine runtime, with UnrealCV enabled. The Python code alone cannot render or run the benchmark.

This Git repository includes task geometry and asset identifiers in `data/`, but excludes Unreal binaries, `.pak` files, marketplace content, model weights, and private run outputs. The compatible legacy Linux build is available from SimWorld's official `Base20260313/Linux.zip`; it already includes the RT-SAFE actors and five levels. See [the pinned download and separate add-on guide](REALTIME_ADDON.md). Do not substitute an arbitrary newer SimWorld build.

## Required runtime contract

- Five levels: `RT10`, `RT12`, `RT15`, `RT18`, `RT20`.
- UnrealCV RPC support, including time control and the methods used by `base/rt_unrealcv.py`.
- Benchmark actor paths from `data/ue_assets.json` and `config.yaml`.
- RT traffic controllers and pedestrian signals under `/Game/RealTimeBench/Traffic/`.
- A Linux launcher and a working GPU/display configuration for the supplied build.

Set `SIMWORLD_UE_LAUNCHER` to the absolute launcher path. Run `python benchmark/run.py doctor smoke` before launching evaluation. The doctor checks configuration, Python modules, launcher presence and model endpoint availability; a successful live smoke run is still required to establish engine compatibility.

The original Linux build's RT-SAFE content is separable as a roughly 36 MB
add-on. See [the real-time add-on guide](REALTIME_ADDON.md) for the upstream
range downloader, packaging tool, checksum-verifying installer, exact base
requirements, and live engine checks. The add-on requires the matching base;
it is not a standalone simulator or a Windows release.

## Before distributing a runtime

See [runtime source and Windows prerequisites](RUNTIME_SOURCE.md) for what is
published, what is missing from an editable engine release, and the validation
needed before supplying a Windows package.

The guide records the verified upstream location, version, hashes, and launch
command. Minimum GPU requirements and Windows/WSL2 compatibility have not been
validated for RT-SAFE. Before mirroring engine assets, establish and include
their third-party notices; the separate fetcher uses SimWorld's own hosting.

Rendered demonstration media is supplied under `website/public/media`. The current cover and film use native Unreal Engine renders of the Madison Square Park NYC scene. The current Astra/Sol comparison uses original first-person RT15 observation and action snapshots, with counts from the recorded logs. The earlier NYC reconstruction is retained as an alternate. The NYC imagery is not the source of the paper tables. See [scene provenance](../website/nyc/README.md).

## NYC transfer pilot

NYC requires a new runtime adapter. It is not a sixth drop-in level for the
legacy UnrealCV launcher. The Madison Square Park project available during
the transfer audit uses Unreal 5.8 and SPEAR. The old cooked RT-SAFE agent,
hazard, manager, and traffic Blueprints do not load in that editor.

The native adapter is implemented in [benchmark/map_transfer](../benchmark/map_transfer/README.md).
The [transfer plan](MAP_TRANSFER.md) explains the migration decisions and the
remaining requirements for a directly comparable benchmark extension.
The workflow follows these stages:

1. **Freeze the old benchmark contract.** Keep RT10/12/15/18/20, their task
   files, and published results unchanged. Record action distances, turning
   angles, observation conventions, inference timing, and safety definitions.
2. **Prepare the new map.** Create an owned derived map; validate collision
   geometry, ground height, spawn clearance, pedestrian corridors, crosswalks,
   and visible signals. Use explicit centimeter coordinates and map-specific
   annotations instead of the old procedural road generator.
3. **Implement native control.** Preserve the 16-action interface. Command
   straight movement directly; automatic navigation must not choose a safer
   route on behalf of the evaluated model. Use a blocking agent capsule.
4. **Instrument safety in the engine.** Record blocking contacts, hazard
   overlap entries, and traffic-rule crossings, including whether they occur
   during inference or action. Deduplicate sustained contacts and distinguish
   goal arrival from arrival with zero safety events.
5. **Validate timing and resets.** Measure simulation time against wall time;
   prove the world pauses during static inference and continues during
   real-time inference. Repeated seeded fixtures must give comparable initial
   states and event traces.
6. **Run small policy pilots.** Start with scripted controls, then a bounded
   number of inexpensive vision-model decisions. Store observations, actions,
   timestamps, event logs, model identity, and configuration with every run.
7. **Report the transfer boundary.** Publish reproducible setup and smoke
   checks. Keep NYC pilot results separate from the original paper tables
   until physics, traffic, hazard, and counting differences have been audited.

The initial runtime audit found two additional issues beyond Blueprint
compatibility: imported mesh collision hulls enclosing empty sidewalks, and
film-rendering settings that advance fixed simulation increments more slowly
than wall time on the test machine. Both require explicit benchmark setup;
a visually correct render does not establish a valid evaluation environment.
