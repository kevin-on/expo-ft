"""Summarize update latency and compilation from a completed benchmark archive."""
import argparse
import json
import re
import statistics
from pathlib import Path


def stats(values):
    if not values:
        return None
    return dict(count=len(values), mean_seconds=statistics.mean(values),
                median_seconds=statistics.median(values), min_seconds=min(values),
                max_seconds=max(values), updates_per_minute=60/statistics.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    events = [json.loads(line) for line in
              (args.output/'small/train-events.jsonl').read_text().splitlines() if line]
    # JAX logging can emit the same compiler message through two handlers.
    compiles = {}
    active = None
    pattern = re.compile(r'Finished XLA compilation of jit\(_update_jit\) in ([0-9.]+) sec')
    with (args.output/'small-train.log').open() as stream:
        for line in stream:
            if line.startswith('[EXPO-GPU] '):
                event = json.loads(line[len('[EXPO-GPU] '):])
                if event['event'] == 'update_start':
                    active = event['update']
                elif event['event'] == 'update_passed':
                    active = None
            match = pattern.search(line)
            if match and active is not None:
                compiles.setdefault(active, set()).add(float(match[1]))
    updates = [dict(update=e['update'], wall_seconds=e['wall_seconds'],
                    xla_compile_seconds=sum(compiles.get(e['update'], ())))
               for e in events if e['event'] == 'update_passed']
    last_compile = max(compiles, default=0)
    # Only call this a post-compilation window if compiler logging was observed.
    stable = [e['wall_seconds'] for e in updates if e['update'] > last_compile] if compiles else []
    memory = [max(e['device_memory'][i].get('peak_bytes_in_use', 0)
                  for e in events)/2**30 for i in range(len(events[0]['devices']))]
    report = dict(runtime=events[0], passed=events[-1]['event']=='passed',
                  updates=updates, last_update_with_xla_compile=last_compile,
                  post_last_compile=stats(stable), last_ten_updates=stats([e['wall_seconds'] for e in updates[-10:]]),
                  jax_peak_GiB_per_device=memory,
                  timing_scope='update() plus block_until_ready; excludes batch preparation and parameter hash checks; includes inference-cache refresh',
                  limitation='Robot-free repeated learner updates; no inference between updates. No kernel-only or robot/network timing.')
    text=json.dumps(report, indent=2)
    (args.output/'speed-summary.json').write_text(text+'\n')
    print(text)


if __name__ == '__main__':
    main()
