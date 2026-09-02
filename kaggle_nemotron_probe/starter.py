import os
import time
import json
import torch
import argparse
import torch.multiprocessing as mp

# Order the task queue by estimated cost ascending instead of alphabetically.
# Unfinished tasks score 0 (coverage is the binding constraint), so finishing
# cheap tasks first maximizes completed-task count for the same wall clock.
# False -> exactly the baseline sorted-key order.
CHEAP_FIRST_ORDER = True

# Read Leg C verified-induction results (if the pre-pass ran) and refund their
# queue slots to the base solver. No results file -> empty skip set -> baseline
# behavior. Must match LEGC_ENABLED in the Leg C launch cell.
LEGC_ENABLED = True


def task_cost(task):
    """Estimated serialized token cost of one task (pure arithmetic).

    Each h x w grid costs h*w digit tokens + h newline/end tokens; a task
    costs the sum over train pair inputs+outputs and test inputs.
    """
    def grid_tokens(grid):
        return len(grid) * len(grid[0]) + len(grid)

    cost = 0
    for pair in task["train"]:
        cost += grid_tokens(pair["input"]) + grid_tokens(pair["output"])
    for pair in task["test"]:
        cost += grid_tokens(pair["input"])
    return cost


def order_keys(data, cheap_first):
    """Deterministic queue order; ties broken by key either way."""
    if not cheap_first:
        return sorted(data.keys())
    return sorted(data.keys(), key=lambda k: (task_cost(data[k]), k))


def fully_verified_task_ids(results):
    """Return tasks safe to remove from the base queue.

    Leg C may verify only a subset of a multi-test task's outputs.  Skipping
    such a task would discard the base solver's predictions for the remaining
    outputs, so queue exclusion requires a non-empty verified entry for every
    output position.
    """
    result = set()
    for task_id, record in results.items():
        if not isinstance(record, dict) or not record.get("verified"):
            continue
        outputs = record.get("outputs")
        if (
            isinstance(outputs, list)
            and outputs
            and all(
                isinstance(output, dict) and output.get("attempt") is not None
                for output in outputs
            )
        ):
            result.add(task_id)
    return result


def local_worker(rank, queue, end_time):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    torch.set_default_device("cpu")

    # Fix Unsloth patching issue
    if rank > 0:
        while not os.path.exists(f"/kaggle/worker{rank-1}"):
            time.sleep(5)
    
    from arc_solver import worker

    with open(f"/kaggle/worker{rank}", "w") as f:
        f.write("Ok")
    
    print(f"[Rank {rank}] start!")
    
    worker(rank, queue, end_time)
    
    print(f"[Rank {rank}] done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    args = parser.parse_args()

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    if rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    with open(test_path, "r") as f:
        data = json.load(f)

    legc_skip = set()
    legc_path = "/kaggle/working/induction_results.json"
    if LEGC_ENABLED and os.path.exists(legc_path):
        with open(legc_path) as f:
            legc_skip = fully_verified_task_ids(json.load(f))
        print(f"[LegC] skipping {len(legc_skip)} verified task(s) in the base queue")

    # Exclude verified tasks BEFORE ordering so the refunded time goes to the
    # cheap-first frontier of the remaining tasks.
    remaining = {k: v for k, v in data.items() if k not in legc_skip}

    queue = mp.Manager().Queue()

    for key in order_keys(remaining, CHEAP_FIRST_ORDER):
        if not rerun_mode:
            if key not in ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]:
                continue
        queue.put(key)
    for _ in range(4):
        queue.put(None)
    
    mp.spawn(local_worker, args=(queue, args.end_time), nprocs=4)
