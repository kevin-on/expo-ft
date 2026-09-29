"""Watch the user-requested allocation and hand it to the validation driver."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
NEW_JOB = "3246733"
OLD_JOB = "3237441"


def run(*args):
    return subprocess.run(args, check=True, universal_newlines=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=45).stdout.strip()


def record(event, **values):
    data = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": event, **values}
    print(json.dumps(data), flush=True)
    temporary = ROOT / "allocation-state.tmp"
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(ROOT / "allocation-state.json")


while True:
    try:
        text = run("scontrol", "show", "job", NEW_JOB, "-o")
        fields = dict(item.split("=", 1) for item in text.split() if "=" in item)
        state = fields["JobState"]
        if state == "RUNNING":
            nodes = run("scontrol", "show", "hostnames", fields["NodeList"]).splitlines()
            if (not fields["UserId"].startswith("kon(") or len(nodes) != 2
                    or "gres/gpu=8" not in fields.get("AllocTRES", "")):
                raise RuntimeError("Allocation is not the expected two-node, eight-GPU job")
            (ROOT / "allocated-nodes.json").write_text(json.dumps({
                "job": NEW_JOB, "nodes": nodes, "job_record": text}, indent=2) + "\n")
            rows = run("squeue", "--noheader", "-u", "kon", "-o", "%i|%u|%T").splitlines()
            old = next((row for row in rows if row.split("|", 1)[0] == OLD_JOB), "")
            if old:
                if not old.startswith(OLD_JOB + "|kon|"):
                    raise RuntimeError("Unexpected owner of previous allocation")
                run("scancel", OLD_JOB)
                record("old_allocation_cancel_requested", new_job=NEW_JOB,
                       old_job=OLD_JOB, nodes=nodes)
            else:
                record("old_allocation_already_absent", new_job=NEW_JOB, nodes=nodes)
            # Preserve the eight-GPU allocation. The separate driver uses srun steps.
            record("allocated_waiting_for_validation_driver", job=NEW_JOB, nodes=nodes)
            while not (ROOT / "VALIDATION_READY").is_file():
                time.sleep(10)
            record("validation_starting", job=NEW_JOB, nodes=nodes)
            result = subprocess.run(["bash", str(ROOT / "run_validation.sh")], cwd=ROOT)
            record("validation_driver_finished", job=NEW_JOB, nodes=nodes,
                   returncode=result.returncode)
            raise SystemExit(result.returncode)
        record("waiting", job=NEW_JOB, state=state, reason=fields.get("Reason"),
               start_time_cluster=fields.get("StartTime"), poll_seconds=600)
        if state not in {"PENDING", "CONFIGURING"}:
            raise SystemExit("Requested allocation ended before validation")
    except (subprocess.SubprocessError, KeyError, OSError) as exc:
        record("poll_error", error=str(exc), job=NEW_JOB)
    time.sleep(600)
