# Release status

This is a clean local release candidate. A remote repository and public Pages site have not yet been created because no GitHub account is authenticated in this environment.

## Included

- Benchmark source, map/task definitions, safety evaluators, model transports and regression tests.
- Static website, manuscript tables, curated UE decision examples, full and short demo films.
- Fresh-repository layout, ignored runtime/output/secrets directories, CI and Pages workflows.
- Installation, protocol, runtime requirements, media provenance and source import manifest.

## Publication dependencies

1. Authenticate GitHub and create the private repository.
2. Select the project-wide code license and supply final author/citation metadata.
3. Provide a compatible Unreal runtime distribution or a reproducible acquisition/build path.
4. Supply the offline BC/RL implementation, dataset and checkpoints if that paper study is to be reproducible from this repository.

The repository can be shared as a partial source release when its scope is stated clearly. It should not be described as a complete end-to-end reproduction package while the runtime and training materials remain unavailable.

Validation details are recorded in `validation/release-checks.json` when checks finish. No paid model calls or live Unreal rollouts are made by release checks.
