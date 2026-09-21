"""Compare single-device update StableHLO, retaining all numerical operations.

Only array-placement and Python output-name metadata is removed. MLIR's standard
inliner/canonicalizer/CSE remove naming and equivalent expression differences.
This does not claim bitwise GPU execution determinism.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

from jax._src.interpreters import mlir
from jaxlib.mlir import ir, passmanager

PLACEMENT_METADATA = {"mhlo.sharding", "jax.result_info", "jax.arg_info"}


def normalize(text):
    with mlir.make_ir_context():
        module = ir.Module.parse(text)

        def clean_attr(value):
            if isinstance(value, ir.DictAttr):
                return ir.DictAttr.get({
                    item.name: clean_attr(item.attr) for item in value
                    if item.name not in PLACEMENT_METADATA
                })
            if isinstance(value, ir.ArrayAttr):
                return ir.ArrayAttr.get([clean_attr(item) for item in value])
            return value

        def visit(op):
            for item in list(op.attributes):
                if item.name in PLACEMENT_METADATA:
                    del op.attributes[item.name]
                else:
                    op.attributes[item.name] = clean_attr(item.attr)
            for region in op.regions:
                for block in region.blocks:
                    for child in block.operations:
                        visit(child.operation)

        visit(module.operation)
        passes = passmanager.PassManager.parse(
            "builtin.module(inline,canonicalize,cse,symbol-dce)")
        passes.run(module.operation)
        return module.operation.get_asm(enable_debug_info=False) + "\n"


def events(root, event):
    return [row for line in (root / "train-events.jsonl").read_text().splitlines()
            if (row := json.loads(line))["event"] == event]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("fixed", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.baseline, args.fixed):
        runtime, = events(path, "runtime")
        assert len(runtime["devices"]) == 1
        assert runtime["batch_size"] == 2 and runtime["utd_ratio"] == 1
    before, after = [events(p, "comparison_input")[0]
                     for p in (args.baseline, args.fixed)]
    input_fields = ("batch_hash", "actor_batch_hash", "learner_rng", "parameter_hashes")
    identical_inputs = all(before[k] == after[k] for k in input_fields)
    captures = [events(p, "lowering_captured")[0] for p in (args.baseline, args.fixed)]
    identical_payload = captures[0]["payload_hash"] == captures[1]["payload_hash"]
    texts = [normalize((p / "update.mlir").read_text())
             for p in (args.baseline, args.fixed)]
    identical_graph = texts[0] == texts[1]
    args.output.mkdir(parents=True, exist_ok=True)
    for label, text in zip(("baseline", "fixed"), texts):
        (args.output / f"{label}.canonical.mlir").write_text(text)
    (args.output / "graph.diff").write_text("".join(difflib.unified_diff(
        texts[0].splitlines(True), texts[1].splitlines(True), "baseline", "fixed")))
    result = dict(
        passed=identical_inputs and identical_payload and identical_graph,
        identical_inputs=identical_inputs,
        identical_full_learner_payload=identical_payload,
        identical_canonical_stablehlo=identical_graph,
        sha256=[hashlib.sha256(text.encode()).hexdigest() for text in texts],
        ignored_metadata=sorted(PLACEMENT_METADATA),
        canonicalization="inline,canonicalize,cse,symbol-dce",
        gpu_execution_determinism_tested=False,
    )
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("Graph or initial-value comparison failed; inspect graph.diff")


if __name__ == "__main__":
    main()
