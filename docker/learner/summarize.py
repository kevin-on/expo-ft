"""Summarize extracted validation results without loading ML libraries."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    report={}
    status=args.output/'run-status.txt'
    if status.exists():report['run_status']=status.read_text().splitlines()
    usage=args.output/'gpu-memory.csv'
    if usage.exists():
        values=[]
        with usage.open() as stream:
            for row in csv.reader(stream):
                try:values.append(int(row[2].strip().split()[0]))
                except (ValueError,IndexError):continue
        if values:report['nvidia_smi_peak_MiB']=max(values)
    for path in sorted(args.output.glob('*/*-events.jsonl')):
        events=[json.loads(line) for line in path.read_text().splitlines() if line]
        peaks=[memory['peak_bytes_in_use'] for event in events for memory in event.get('device_memory',[])
               if 'peak_bytes_in_use' in memory]
        phases=dict(last_event=events[-1]['event'],passed=events[-1]['event']=='passed',
                    runtime=events[0],timings=[{k:v for k,v in e.items() if k!='device_memory'}
                    for e in events if e['event'] in ['inference','update_passed']])
        if peaks:phases['jax_peak_GiB']=max(peaks)/2**30
        report[str(path.relative_to(args.output))]=phases
    text=json.dumps(report,indent=2)
    (args.output/'summary.json').write_text(text+'\n')
    print(text)


if __name__=='__main__':
    main()
