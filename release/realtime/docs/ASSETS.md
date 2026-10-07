# Runtime assets

Use the exact legacy Linux build and hashes in [REALTIME_ADDON.md](REALTIME_ADDON.md).
The runtime is hosted publicly by SimWorld. The full archive includes the
real-time chunk; the small fetch tool also obtains that chunk independently.
UnrealCV is compiled into the engine executable. A matching Unreal version
alone does not establish compatibility; use the installer fingerprints and
live five-map checker before evaluating models.

Model weights are not included. Hosted models use their own authenticated
API or CLI transport. Local models need their own serving environment.
