#!/usr/bin/env bash
# Start a packaged-UE fleet and keep it alive until Ctrl-C/SIGTERM.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
: "${UE_LAUNCHER:?set UE_LAUNCHER to the packaged SimWorld.sh}"

read -r -a GPU_ARGS <<< "${UE_GPUS:-0 1 2 3}"
read -r -a PORT_ARGS <<< "${UE_PORTS:-9010 9020 9030 9040}"
ENDPOINTS=${SIMWORLD_RL_UE_ENDPOINTS:-/tmp/simworld-rt-endpoints.json}
UE_LOG_DIR=${UE_LOG_DIR:-/tmp/simworld-rt-ue-logs}

if [[ ${#GPU_ARGS[@]} -ne ${#PORT_ARGS[@]} ]]; then
  echo "UE_GPUS and UE_PORTS must have the same number of entries" >&2
  exit 2
fi

echo "Starting ${#GPU_ARGS[@]} RT10 UE instance(s). Keep this terminal open."
echo "Endpoint manifest: ${ENDPOINTS}"
exec "${PYTHON_BIN}" "${REPO}/online_rl/launch_ue_fleet.py" \
  --launcher "${UE_LAUNCHER}" \
  --map "${UE_MAP:-RT10}" \
  --gpus "${GPU_ARGS[@]}" \
  --ports "${PORT_ARGS[@]}" \
  --output "${ENDPOINTS}" \
  --logs "${UE_LOG_DIR}"
