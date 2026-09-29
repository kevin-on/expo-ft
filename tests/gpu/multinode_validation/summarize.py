"""Summarize same-clock 10Hz WAN cycle timings; no model imports."""
import argparse
import json
from pathlib import Path
from statistics import median


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learner', type=Path, required=True)
    parser.add_argument('--inference', type=Path, required=True)
    parser.add_argument('--first-round', type=int, default=2)
    parser.add_argument('--cycles', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rollout = {item['round']: item for item in rows(args.inference / 'rollout-timing.jsonl')}
    metrics = rows(args.learner / 'metrics.jsonl')
    updates = {item['episodes'] // 2 - 1: item for item in metrics if 'split/update_seconds' in item}
    ack = {item['step']: item for item in metrics if 'split/through_inference_ready_seconds' in item}
    saves = rows(args.learner / 'replay-save-timing.jsonl')
    output = []
    for index in range(args.first_round, args.first_round + args.cycles):
        current, following = rollout[index], rollout[index + 1]
        update = updates[index]
        policy = ack[update['step']]
        cycle = dict(round=index, full_iteration=following['start'] - current['start'],
                     rollout=current['end'] - current['start'],
                     episode_end_to_next_start=following['start'] - current['end'],
                     replay_save=sum(row['seconds'] for row in saves[2 * index:2 * index + 2]),
                     updates=update['split/update_seconds'])
        for key in ('export_seconds', 'hash_handoff_seconds', 'receive_seconds',
                    'verify_seconds', 'install_seconds', 'through_inference_ready_seconds'):
            cycle[key] = policy['split/' + key]
        cycle['other_episode_end_overhead'] = (cycle['episode_end_to_next_start'] -
            cycle['updates'] - cycle['through_inference_ready_seconds'])
        output.append(cycle)
    result = dict(cycles=output, medians={key: median(row[key] for row in output)
                                        for key in output[0] if key != 'round'})
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['medians'], indent=2))


if __name__ == '__main__':
    main()
