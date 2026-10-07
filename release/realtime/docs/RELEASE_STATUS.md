# Real-time release scope

This source distribution covers the original five-map benchmark only. It
excludes the website, map-transfer implementation, Unreal binaries, generated
campaigns and credentials. `SOURCE_MANIFEST.json` records each shipped file
and the originating development commit.

Fresh-install evidence and example comparisons are linked from
[examples](../examples/README.md). Runtime checks test rendering, Blueprint
availability, state APIs and pause/resume timing. They do not establish exact
historical model scores, native Windows/WSL2 support or minimum GPU memory.

The original code license and editable Unreal project are not supplied by
this release. Third-party engine/content redistribution remains upstream.
