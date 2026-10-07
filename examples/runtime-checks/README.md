# Fresh runtime screenshots

These unedited frames were produced by the model-free runtime checker after
downloading the pinned upstream ZIP, verifying its full SHA-256, extracting
it with Python, and reinstalling the independently downloaded real-time chunk.
A new Python environment and cache directories were used on Linux / RTX A5000.

| Map | Paused movement | Resumed movement | Render |
|---|---:|---:|---|
| RT10 | 0.0 cm | 231.2 cm | [PNG](RT10.png) |
| RT12 | 0.0 cm | 162.5 cm | [PNG](RT12.png) |
| RT15 | 0.0 cm | 168.1 cm | [PNG](RT15.png) |
| RT18 | 0.0 cm | 323.1 cm | [PNG](RT18.png) |
| RT20 | 0.0 cm | 41.4 cm | [PNG](RT20.png) |

The checker requires no paused drift and positive resumed movement; exact
distances and pixels depend on physics/render scheduling and are not score
targets. It also verifies all required Blueprint classes and state fields.
See the [full validation record](../../validation/realtime-fresh-runtime.json).
