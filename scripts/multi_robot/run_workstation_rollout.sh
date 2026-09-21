#!/bin/bash
# iris-ws-5 workstation launcher; paths/device mappings are local to this setup.
# iris6 and ports 8102/8103 below belong to the original learner allocation.
# Update that endpoint when using a different GPU server.
# Starts real hardware when the learner requests create_env/reset.
set -euo pipefail
robot_index=${1:?Usage: run_workstation_rollout.sh 0|1}
case "$robot_index" in 0|1) ;; *) echo 'Robot index must be 0 or 1' >&2; exit 2;; esac
source /scr/kevinon/env.sh
cd /scr/kevinon/workspace/expo-ft-fork
# Refuse the temporary blank-camera setup for this real four-camera run.
client/.venv/bin/python - "$robot_index" <<'PY'
import json
import fcntl
import os
import tempfile
import sys
from pathlib import Path
import pyzed.sl as sl
import pyspacemouse as sm
# SDK enumeration probes devices; serialize simultaneous client preflights.
with open(os.path.join(tempfile.gettempdir(), f'droid-zed-init-{os.getuid()}.lock'), 'a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    available={str(d.serial_number) for d in sl.Camera.get_device_list() if d.camera_state==sl.CAMERA_STATE.AVAILABLE}
robot_index=int(sys.argv[1])
space_paths={d.path for d in sm.Enumeration().find() if 'SpaceMouse' in (d.product_string or '')}
owned=set()
for i in (0,1):
    cfg=json.loads(Path(f'configs/robots/robot-{i}.json').read_text())
    serials=set(cfg['camera_serials'])
    if cfg.get('blank_camera_serials') or any(s.startswith('TEMP_') for s in serials):
        raise SystemExit(f'Robot {i}: replace temporary camera mapping before real rollout')
    if serials & owned:
        raise SystemExit('Camera ownership overlaps between robots')
    owned |= serials
    missing=serials-available
    if i == robot_index and missing:
        raise SystemExit(f'Robot {i}: cameras not AVAILABLE: {sorted(missing)}')
    if cfg['spacemouse_device_path'] not in space_paths:
        raise SystemExit(f'Robot {i}: configured SpaceMouse path missing')
print(f'Robot {robot_index}: cameras AVAILABLE; both robot mappings and SpaceMouse enumeration passed')
PY
exec client/.venv/bin/python -m client.run_client \
 --host iris6.stanford.edu --port "$((8102 + robot_index))" \
 --config-task-path configs/task/pick.py --robot-config "configs/robots/robot-${robot_index}.json"
