# Unreal runtime and assets

Live evaluation requires the RT-SAFE-compatible SimWorld Unreal Engine runtime, with UnrealCV enabled. The Python code alone cannot render or run the benchmark.

The release includes task geometry and asset identifiers in `data/`, but excludes Unreal binaries, `.pak` files, marketplace content, model weights, and private run outputs. A redistributable runtime download has not yet been supplied. Obtain the compatible runtime from the project maintainers; do not assume a stock SimWorld build contains the RT-SAFE actors.

## Required runtime contract

- Five levels: `RT10`, `RT12`, `RT15`, `RT18`, `RT20`.
- UnrealCV RPC support, including time control and the methods used by `base/rt_unrealcv.py`.
- Benchmark actor paths from `data/ue_assets.json` and `config.yaml`.
- RT traffic controllers and pedestrian signals under `/Game/RealTimeBench/Traffic/`.
- A Linux launcher and a working GPU/display configuration for the supplied build.

Set `SIMWORLD_UE_LAUNCHER` to the absolute launcher path. Run `python benchmark/run.py doctor smoke` before launching evaluation. The doctor checks configuration, Python modules, launcher presence and model endpoint availability; a successful live smoke run is still required to establish engine compatibility.

## Before distributing a runtime

Record the runtime version, download location, SHA-256, Unreal version, GPU requirements, launch command, and applicable third-party asset terms. The currently available workspace does not establish all of these, so no download or compatibility claim is invented here.

Rendered demonstration media is supplied separately under `website/public/media`. The generated cover is labeled as a concept illustration. Pilot footage illustrates the interface and is not the source of the paper tables.
