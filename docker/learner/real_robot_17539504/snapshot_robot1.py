"""Bundle committed replay files and the latest complete checkpoint as one file."""
import json
import os
from pathlib import Path
import tarfile

stage=Path('/tmp/kevinon/expo-real-17539504')
out=stage/'output'
ckpt=out/'pick-robot1-a40x2-b8-utd20-s42-j17539504/checkpoints'
remote=Path('/iris/u/kevinon/outputs/expo-ft-real/17539504-robot1')
remote.mkdir(parents=True,exist_ok=True)
records=sorted(ckpt.glob('buffers/*.pkl'))
steps=sorted(int(p.name) for p in ckpt.iterdir() if p.is_dir() and p.name.isdigit()) if ckpt.exists() else []
latest=steps[-1] if steps else None
if not records and latest is None:
    raise SystemExit(0)
# Replay writes are atomic (temporary file + rename), so listed .pkl files are complete.
# Extra replay after the model step is retained; resume discards the suffix.
manifest={'model_step':latest,'replay_records':len(records),'max_replay_step':max((int(p.stem) for p in records),default=0)}
marker=stage/'last-snapshot-robot1.json'
if marker.exists() and json.loads(marker.read_text())==manifest:
    raise SystemExit(0)
archive=remote/'recovery.tar.partial'
with tarfile.open(archive,'w') as tar:
    for p in records:
        tar.add(p,arcname=str(p.relative_to(out)),recursive=False)
    if latest is not None:
        for p in [ckpt/str(latest)]:
            tar.add(p,arcname=str(p.relative_to(out)))
    for p in [ckpt/'wandb_id.txt',stage/'source-manifest.json']:
        if p.exists(): tar.add(p,arcname=p.name)
os.replace(archive,remote/'recovery.tar')
marker.write_text(json.dumps(manifest)+'\n')
(remote/'recovery.json').write_text(json.dumps(manifest)+'\n')
print('Saved recovery snapshot',manifest,flush=True)
