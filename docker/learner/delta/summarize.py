"""Summarize completed or interrupted Delta validation without hiding failures."""
import argparse
import json
import re
import statistics
from pathlib import Path


def events(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A killed writer can leave an incomplete final event.
            continue
    return rows


def summarize_case(root, label):
    train = events(root / label / 'train-events.jsonl')
    restore = events(root / label / 'restore-events.jsonl')
    runtime = next((r for r in train if r['event'] == 'runtime'), {})
    compiles = {}
    active = None
    pattern = re.compile(r'Finished XLA compilation of jit\(_update_jit\) in ([0-9.]+) sec')
    log = root / f'{label}-train.log'
    for line in log.read_text().splitlines() if log.exists() else []:
        if line.startswith('[EXPO-GPU] '):
            try:
                event = json.loads(line[len('[EXPO-GPU] '):])
            except json.JSONDecodeError:
                continue
            if event['event'] == 'update_start':
                active = event['update']
            elif event['event'] == 'update_passed':
                active = None
        match = pattern.search(line)
        if match and active is not None:
            # The JAX message can appear twice through different logging handlers.
            compiles.setdefault(active, set()).add(float(match[1]))
    updates = [dict(update=r['update'], seconds=r['wall_seconds'],
                    compile_seconds=sum(compiles.get(r['update'], ())))
               for r in train if r['event'] == 'update_passed']
    last_compile = max(compiles, default=None)
    # A one-update smoke test cannot establish steady-state speed.
    steady = [r['seconds'] for r in updates if r['update'] > last_compile] if compiles else []
    train_passed = bool(train and train[-1]['event'] == 'passed'
                        and train[-1].get('checkpoint_saved')
                        and len(updates) == runtime.get('updates'))
    restore_passed = bool(restore and restore[-1]['event'] == 'passed')
    memory = {}
    for row in train + restore:
        for index, device in enumerate(row.get('device_memory', [])):
            memory[index] = max(memory.get(index, 0), device.get('peak_bytes_in_use', 0))
    return dict(runtime=runtime, train_passed=train_passed, restore_passed=restore_passed,
                passed=train_passed and restore_passed, updates=updates,
                last_compile_update=last_compile, steady_update_count=len(steady),
                steady_mean_seconds=statistics.mean(steady) if steady else None,
                steady_min_seconds=min(steady) if steady else None,
                steady_max_seconds=max(steady) if steady else None,
                jax_peak_GiB_per_device={str(k): v / 2**30 for k, v in memory.items()},
                inference=[dict(label=r['label'], seconds=r['wall_seconds'])
                           for r in train if r['event'] == 'inference'])


def summarize(root):
    settings = {}
    settings_path = root / 'settings.txt'
    if settings_path.exists():
        settings = dict(line.split('=', 1) for line in settings_path.read_text().splitlines() if '=' in line)
    # Include requested cases even if execution failed before creating their files.
    labels = {f'batch-{batch}' for batch in settings.get('batch_sizes', '').split()}
    labels.update(p.name for p in root.glob('batch-*') if p.is_dir())
    cases = {label: summarize_case(root, label) for label in sorted(labels)}
    peaks = {}
    gpu_log = root / 'gpu-memory.csv'
    for line in gpu_log.read_text().splitlines() if gpu_log.exists() else []:
        fields = line.split(',')
        try:
            uuid, mib = fields[1].strip(), float(fields[2].split()[0])
            peaks[uuid] = max(peaks.get(uuid, 0), mib)
        except (IndexError, ValueError):
            continue
    return dict(passed=bool(cases) and all(c['passed'] for c in cases.values()),
                settings=settings, cases=cases, nvidia_smi_peak_MiB_per_device=peaks,
                timing_scope='update plus block_until_ready, including inference-cache refresh; excludes batch preparation and parameter hashing',
                limitation='Robot-free repeated updates; no inference between updates. GPU memory sampling includes allocator reservation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = summarize(args.output)
    text = json.dumps(report, indent=2) + '\n'
    (args.output / 'summary.json').write_text(text)
    print(text, end='')
    raise SystemExit(0 if report['passed'] else 1)
