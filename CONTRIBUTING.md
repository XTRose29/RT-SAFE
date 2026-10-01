# Contributing

Use Python 3.12 and Node 22. Run `python -m pytest -q` from the root and `npm test`, `npm run typecheck`, `npm run build` from `website/` before opening a pull request. Describe any changes to prompts, timing, safety rules or task geometry: they can change the benchmark being measured.

Keep credentials, runtime assets, model weights and raw campaign output out of commits. Add only curated, documented result artifacts. Report the model ID, transport, reasoning configuration, seed, runtime version and resolved suite config with a new result.

The manuscript tables are a versioned research snapshot. Do not replace them with unrelated pilot results.
