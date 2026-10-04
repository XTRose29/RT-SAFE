# Release status

This is a private-first release candidate. The remote repository and Pages URL are established during publishing and are not hardcoded in the source.

## Included

- Benchmark source, map/task definitions, safety evaluators, model transports and regression tests.
- Historical VAGEN online-RL pipeline, restored from research commit `1bbb87de`, with adapter/fleet tests. Live training remains unvalidated.
- Static website, manuscript tables, native NYC scene footage, a 90-second narrated film, and a matched Astra/Sol replay with source evidence.
- Fresh-repository layout, ignored runtime/output/secrets directories, CI and Pages workflows.
- Installation, protocol, runtime requirements, media provenance and source import manifest.

## Publication dependencies

1. Authenticate GitHub and create the private repository.
2. Select the project-wide code license and supply final author/citation metadata.
3. Provide a compatible Unreal runtime distribution or a reproducible acquisition/build path.
4. Supply the offline BC/RL implementation, dataset and checkpoints if that paper study is to be reproducible from this repository.

The repository can be shared as a partial source release when its scope is stated clearly. It should not be described as a complete end-to-end reproduction package while the runtime and training materials remain unavailable.

The benchmark tests pass (684 tests and 407 subtests; one optional serving module skipped). The website passes its six data tests, TypeScript check, production build, local asset/link checks and Pages subpath/metadata checks.

Validation details are recorded in [validation/release-checks.json](../validation/release-checks.json). No paid model calls or live Unreal rollouts are made by release checks.

## NYC presentation edition

The 4 October 2026 edition adds native Madison Square Park rendering, four camera views,
a synchronized static/real-time illustration, and the recorded Astra/Sol segment reconstructed
in NYC. The narrated and captioned films are both exactly 90 seconds (2,700 frames at 30 fps).
The full selected comparison is 44.1 seconds at 6× simulation speed.

The native renders completed successfully. Both film editions and the full comparison decode
without errors. Narration measures −16.07 LUFS integrated and −1.48 dBTP true peak. The
website passes desktop/mobile playback, seek/replay, camera-gallery, reduced-motion, and
layout checks. No browser errors were reported. The web aerial loop is an 8 MB 1080p encode;
the higher-quality render and intermediate frames remain in the local production workspace.

See [NYC validation](../validation/nyc-release-checks.json) and
[rendering provenance](../website/nyc/README.md). The earlier benchmark test results above
remain from the prior release check; this edition does not change benchmark execution code.
