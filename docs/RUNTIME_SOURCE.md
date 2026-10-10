# Runtime source and Windows prerequisites

The original five-map benchmark can be run with the publicly hosted Linux
runtime. The published Python source and the cooked runtime do not constitute
an editable Unreal project. Use the [pinned download and live checker](REALTIME_ADDON.md)
for the supported installation.

## What is available

| Material | Availability and purpose |
| --- | --- |
| Python benchmark, evaluation and tests | Included in this repository. |
| Five-map layout and task data | `data/map1_10roads/` through `data/map5_20roads/`: roads, buildings, scene placements, obstacles and routes. |
| Unreal asset identifiers | `data/ue_assets.json`, `config.yaml`, and the Python runtime code reference required Blueprint classes. These identifiers are not the assets themselves. |
| Original Linux engine and RT content | Upstream `Base20260313/Linux.zip`, including `pakchunk3002-Linux.pak`; download, hashes and validation are documented in the runtime guide. |
| Editable original RT maps and Blueprints | Not included in this release. A complete uncooked source project has not been located in the repository/source audit. |
| Windows RT-SAFE package | Not currently supplied or validated. |

Scene placement data can help reconstruct a layout, but it does not preserve
all Blueprint behavior, collision setup, navigation, materials, or engine
settings. A Python call that spawns `/Game/RealTimeBench/...` assumes that
class already exists in Unreal. It does not recreate the Blueprint source.
Extracting `.umap` or `.uasset` files from a cooked Linux package does not
restore editor data or provide a source project for a Windows cook.

## Requirements for an original Windows package

SimWorld has a Windows base and documents
[Windows content packaging](https://github.com/SimWorld-AI/SimWorld/blob/main/docs/source/customization/make_your_own_pak.rst).
That establishes a possible build route; it does not establish RT-SAFE compatibility.

Before promising an original-five-map Windows release, obtain and verify:

1. The uncooked RT10, RT12, RT15, RT18 and RT20 maps, agent and hazard
   Blueprints, traffic controllers/signals, and all base asset dependencies.
2. The matching UE 5.3.2 project configuration, including collision, physics,
   navigation and packaging settings. A newer city project is not an equivalent
   replacement for the original benchmark.
3. A Windows executable with compatible UnrealCV behavior. Test the existing
   Windows base first; rebuilding the executable is necessary only if its
   required interfaces or dependencies are incompatible. A rebuild needs
   compatible plugin/project source and a Windows build environment.
4. A Windows cook of the RT content and any required base dependencies.
   Merely copying or renaming `pakchunk3002-Linux.pak` is not a validated port.
5. Appropriate source/binary distribution terms and third-party notices.

Validate all five maps, nonblank rendering, actor spawning/state APIs,
pause/resume movement, resets, and representative hazard/traffic behavior.
Record executable/content hashes, engine and plugin versions, hardware,
commands and test logs. Basic smoke checks alone do not establish equality
with historical benchmark scores.

The current `tools/check_runtime_addon.py` launches and owns a local Linux
engine process. It has no attach-to-existing-engine mode. A Windows engine
controlled from WSL needs a separate validation path or an explicit extension
to the checker. Python runners that accept `--ue-ip` and `--ue-port` can target
a separately launched engine, but that does not establish Windows compatibility.
WSL-to-Windows `localhost` depends on the networking configuration; see
[Microsoft's WSL networking guide](https://learn.microsoft.com/en-us/windows/wsl/networking).

## Publication boundary

Original RT-SAFE code is currently source-available; a project-wide license
is still pending the authors' selection. See [third-party notices](../THIRD_PARTY_NOTICES.md).
Existing third-party licenses remain in force. Do not describe the whole
Unreal runtime as open source or infer redistribution permission from its
public download URL. Continue distributing the verified upstream download
instructions while editable assets, build provenance and licensing are resolved.
