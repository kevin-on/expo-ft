"""Send eval-only weights through an already running RAM transport/SSH relay."""
import argparse
import os
from pathlib import Path
import uuid


def main():
    # Export is host-only; do not consume a learner GPU or change its process.
    os.environ['JAX_PLATFORMS']='cpu'
    from expo_ft.distributed.channel import Channel
    from .checkpoint import export, manifest, read_file, within, relative_path, identifier
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mailbox',required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint',type=Path)
    source.add_argument('--weights',type=Path,help='Saved trainable_weights.bin or checkpoint directory (legacy eval/weights.bin accepted)')
    p.add_argument('--kind',choices=['sft','online'],default='sft')
    p.add_argument('--checkpoint-path',help='Checkpoint path relative to the experiments root; preserved verbatim')
    p.add_argument('--training-run-id',help='Registered training run ID')
    source.add_argument('--experiments-root',type=Path,help='Local root; read checkpoint-path under this root instead of --checkpoint')
    p.add_argument('--initial-sft-checkpoint',type=Path)
    p.add_argument('--replan-steps',type=int,default=8)
    p.add_argument('--timeout',type=float,default=600)
    a=p.parse_args()
    if not a.weights:
        if not a.checkpoint_path or not a.training_run_id:
            p.error('--checkpoint-path and --training-run-id are required when exporting')
        relative_path(a.checkpoint_path);identifier(a.training_run_id)
        if a.experiments_root: a.checkpoint=within(a.experiments_root,a.checkpoint_path)
    elif a.checkpoint_path or a.training_run_id:
        p.error('--weights already contains its checkpoint path and training run ID')
    c=Channel(a.mailbox,timeout=a.timeout)
    request=uuid.uuid4().hex
    try:
        # Admission BEFORE expensive checkpoint disk reads. Receiver reserves its idle mode.
        c.send('eval-offer',request,{'timeout':a.timeout});c.flush()
        reply=c.receive('eval-admission',request);c.release('eval-admission',request)
        if not reply['accepted']: raise RuntimeError(reply['reason'])
        try:
            print('Exporting checkpoint into CPU RAM...',flush=True)
            payload=read_file(a.weights) if a.weights else export(a.checkpoint,a.kind,
                checkpoint_path=a.checkpoint_path,training_run_id=a.training_run_id,
                initial_sft=a.initial_sft_checkpoint,replan_steps=a.replan_steps)
            with payload:
                info=manifest(payload)['metadata']
                print(f"Sending {info['checkpoint_path']}: {payload.size/1e9:.3f} GB",flush=True)
                c.send_buffer('eval-weights',request,payload)
                c.wait_sent('eval-weights',request)
                result=c.receive('eval-result',request);c.release('eval-result',request)
                if not result['accepted']: raise RuntimeError(result['reason'])
                print('Received and validated on eval server (RAM only).',flush=True)
        except BaseException:
            c.send('eval-cancel',request,{});c.flush()
            raise
    finally: c.close()


if __name__=='__main__':main()
