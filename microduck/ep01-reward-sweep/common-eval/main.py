"""Meshive Task: re-score every Microduck Sprint candidate in one common CPU environment.

Upload as main.py. Use Python 3.12 with a CPU preset; Arena scoring uses no GPU.
Attach the five training Task outputs as input assets (see common-eval-guide.md).
No remote publication, Hub upload, or cloud resource creation is performed here.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# The evaluator the training Tasks used: microduck-challenges uv.lock at 9aaec6db pins this wheel and hash.
ARENA_WHEEL_URL = ("https://github.com/pollen-robotics/microduck-arena/releases/download/"
                   "v0.5.1/microduck_arena-0.5.1-py3-none-any.whl")
ARENA_WHEEL_SHA256 = "783049640bea382af3dfa0075844c3282152ba1a587b48678c6c81f965855eaf"
PINNED = {"microduck-arena": "0.5.1", "mujoco": "3.10.0", "numpy": "2.4.1", "onnxruntime": "1.24.4"}
PROBE = ("import importlib.metadata as m, json, arena.cli; "
         f"print(json.dumps({{name: m.version(name) for name in {sorted(PINNED)!r}}}))")
EVENT = "sprint-2m"
# Everything that must match across inputs; only forward_reward may differ.
CONDITIONS = ("mode", "task", "event", "repository", "base_revision", "num_envs", "max_iterations", "seed")
RECORDED = re.compile(r"^recorded (\S+)$", re.M)
SHA256_NAME = re.compile(r"[0-9a-f]{64}")
COLUMNS = ["rank", "role", "forward_reward", "checkpoint", "task_score", "task_seeds_finished",
           "common_score", "common_seeds_finished", "seeds_total", "score_change_s", "command_vx",
           "run_id", "model_sha256"]


def available_cpus(cgroup=Path("/sys/fs/cgroup")):
    """CPUs this container may use. A Task's CPU limit is a cgroup quota; os.cpu_count() shows the host."""
    try:
        quota, period = (cgroup / "cpu.max").read_text().split()  # v2: "400000 100000" or "max 100000"
    except (OSError, ValueError):
        try:
            quota = (cgroup / "cpu/cpu.cfs_quota_us").read_text().strip()  # v1: -1 means no limit
            period = (cgroup / "cpu/cpu.cfs_period_us").read_text().strip()
        except OSError:
            quota = period = None
    if quota not in (None, "max", "-1"):
        return max(1, int(quota) // int(period))
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def settings(environ):
    cfg = {
        "input_root": environ.get("INPUT_ROOT", "/inputs"),
        "candidates": environ.get("CANDIDATES", "all"),
        "workers": int(environ.get("WORKERS") or available_cpus()),
        "expected_tasks": int(environ.get("EXPECTED_TASKS", "5")),
        "event": EVENT,
        "arena_wheel": ARENA_WHEEL_URL,
    }
    if cfg["candidates"] not in {"all", "best"}:
        raise ValueError("CANDIDATES must be all or best")
    if not 1 <= cfg["workers"] <= 128:
        raise ValueError("WORKERS must be between 1 and 128")
    if not 1 <= cfg["expected_tasks"] <= 20:
        raise ValueError("EXPECTED_TASKS must be between 1 and 20")
    return cfg


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank_key(row, prefix=""):
    """Most seeds finished first, then lowest mean time; ties keep the smaller reward and checkpoint."""
    return (-row[prefix + "seeds_finished"], row[prefix + "score"], row["forward_reward"], row["checkpoint"])


def discover(root):
    """Training Task outputs anywhere under root: results.json beside settings.json and status.json."""
    if not root.is_dir():
        raise FileNotFoundError(f"Input root not found: {root}")
    tasks = []
    for results in sorted(root.rglob("results.json")):
        folder = results.parent
        for name in ["settings.json", "status.json"]:
            if not (folder / name).is_file():
                raise ValueError(f"{results} has no {name} beside it; attach it as well")
        state = read_json(folder / "status.json").get("state")
        if state != "complete":
            raise ValueError(f"{folder}: training Task state is {state!r}, not complete")
        rows = read_json(results)
        if not rows:
            raise ValueError(f"{results} has no evaluated checkpoints")
        reference = folder / "reference.json"
        tasks.append({"dir": folder, "settings": read_json(folder / "settings.json"), "rows": rows,
                      "reference": read_json(reference) if reference.is_file() else None})
    if not tasks:
        raise FileNotFoundError(f"No training Task outputs (results.json + settings.json) under {root}")
    return tasks


def check_inputs(tasks, expected):
    base = tasks[0]
    rewards = [task["settings"]["forward_reward"] for task in tasks]
    repeated = sorted({reward for reward in rewards if rewards.count(reward) > 1})
    if repeated:
        raise ValueError(f"Duplicate inputs for forward reward {repeated}; attach each Task once")
    for task in tasks[1:]:
        changed = [key for key in CONDITIONS if task["settings"].get(key) != base["settings"].get(key)]
        if changed:
            raise ValueError(f"Mixed training conditions ({', '.join(changed)}): {task['dir']} vs {base['dir']}")
    if len(tasks) != expected:
        raise ValueError(f"Expected {expected} Task outputs, found {len(tasks)} (rewards {sorted(rewards)}); "
                         "attach the missing ones or set EXPECTED_TASKS")
    references = {task["reference"]["model_sha256"] if task["reference"] else None for task in tasks}
    if len(references) > 1:
        raise ValueError("Inputs disagree on the reference model (reference.json missing or different)")
    return {key: base["settings"].get(key) for key in CONDITIONS}, references.pop()


def index_models(root):
    """sha256 -> path for every ONNX under root, wherever the attachment placed it."""
    models = {}
    for path in sorted(root.rglob("*.onnx")):
        digest = sha256_file(path)
        if SHA256_NAME.fullmatch(path.stem) and path.stem != digest:
            raise ValueError(f"Corrupt model: {path} does not hash to its name")
        models.setdefault(digest, path)
    return models


def plan(cfg):
    root = Path(cfg["input_root"])
    tasks = discover(root)
    conditions, reference_sha = check_inputs(tasks, cfg["expected_tasks"])
    candidates = []
    for task in tasks:
        reward = task["settings"]["forward_reward"]
        rows = task["rows"] if cfg["candidates"] == "all" else [min(task["rows"], key=rank_key)]
        candidates += [{"forward_reward": reward, "checkpoint": row["checkpoint"],
                        "model_sha256": row["model_sha256"], "task_score": row["score"],
                        "task_seeds_finished": row["seeds_finished"]} for row in rows]
    names = {reference_sha: "reference-common"} if reference_sha else {}
    for candidate in candidates:  # One score per distinct model, even if two rows share it.
        names.setdefault(candidate["model_sha256"],
                         f"common-forward{candidate['forward_reward']:g}-{candidate['checkpoint']}")
    models = index_models(root)
    missing = [name for sha, name in names.items() if sha not in models]
    if missing:
        raise FileNotFoundError(f"No ONNX under {root} for {', '.join(missing)}; "
                                "attach each Task's arena/models folder")
    return {"tasks": tasks, "conditions": conditions, "reference_sha": reference_sha,
            "candidates": candidates, "jobs": {sha: (name, models[sha]) for sha, name in names.items()}}


def describe(work_plan):
    return {"conditions": work_plan["conditions"], "reference_sha256": work_plan["reference_sha"],
            "tasks": [{"dir": str(task["dir"]), "forward_reward": task["settings"]["forward_reward"],
                       "checkpoints": [row["checkpoint"] for row in task["rows"]]}
                      for task in work_plan["tasks"]],
            "candidates": len(work_plan["candidates"]), "models_to_score": len(work_plan["jobs"])}


def run(command, logfile, env, cwd=None, echo=True):
    command = list(map(str, command))
    if echo:
        print("$ " + " ".join(command), flush=True)
    lines = []
    with logfile.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace", bufsize=1)
        try:
            for line in process.stdout:
                if echo:
                    print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
                lines.append(line)
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        if not echo:
            print(f"--- {logfile.name} (last lines)\n" + "".join(lines[-20:]), end="", flush=True)
        raise subprocess.CalledProcessError(code, command)
    return "".join(lines)


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def execute(cfg, out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "settings.json").exists():
        raise RuntimeError("Use a fresh Task output directory; settings.json already exists")
    write_json(out / "settings.json", cfg)
    logs = out / "evaluation"
    logs.mkdir()
    store = out / "arena"
    work = Path(tempfile.mkdtemp(prefix="meshive-common-eval-"))
    # Each arena score is one thread (onnxruntime intra-op 1, one MuJoCo loop); keep BLAS at one too.
    env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")

    def phase(name, **extra):
        print(f"\n[MESHIVE] {name}", flush=True)
        write_json(out / "status.json", {"phase": name, "state": "running", **extra})

    def score(sha, name, onnx):
        output = run([sys.executable, "-m", "arena.cli", "score", onnx, "--event", EVENT, "--user", "meshive",
                      "--name", name, "--store", store], logs / f"score-{name}.log", env, cwd=work, echo=False)
        run_ids = RECORDED.findall(output)  # Parallel runs share the store, so never diff the directory.
        if len(run_ids) != 1:
            raise RuntimeError(f"Expected one 'recorded <id>' line for {name}, got {run_ids}")
        record = read_json(store / "runs" / EVENT / f"{run_ids[0]}.json")
        if record["model_sha256"] != sha:
            raise RuntimeError(f"Record {run_ids[0]} does not belong to {name}")
        return record

    try:
        if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
            raise RuntimeError("This Task requires Linux with Python 3.12")
        phase("SETUP")
        work_plan = plan(cfg)
        jobs = work_plan["jobs"]
        write_json(out / "inputs.json", describe(work_plan))
        print(json.dumps(describe(work_plan), indent=2), flush=True)

        phase("INSTALL")
        wheel_dir = work / "wheel"
        run([sys.executable, "-m", "pip", "download", "--no-deps", "--no-cache-dir", "--dest", wheel_dir,
             ARENA_WHEEL_URL], logs / "wheel-download.log", env)
        wheel = wheel_dir / ARENA_WHEEL_URL.rsplit("/", 1)[1]
        if not wheel.is_file() or sha256_file(wheel) != ARENA_WHEEL_SHA256:
            raise RuntimeError("Arena wheel is missing or does not match the training lockfile sha256")
        run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--disable-pip-version-check", wheel],
            logs / "install.log", env)
        run([sys.executable, "-m", "pip", "freeze"], logs / "pip-freeze.txt", env, echo=False)
        packages = json.loads(run([sys.executable, "-c", PROBE], logs / "import-check.log", env,
                                  cwd=work).strip().splitlines()[-1])
        if packages != PINNED:
            raise RuntimeError(f"Evaluator packages differ from the pins: {packages}")

        phase("CHECK")  # Arena's own physics tripwire: the reference walker on this OS/CPU architecture.
        golden = re.search(r"golden run holds: ([\d.]+)s",
                           run([sys.executable, "-m", "arena.golden"], logs / "golden.log", env, cwd=work))
        environment = {"machine": platform.machine(), "platform": platform.platform(), "cpu_model": cpu_model(),
                       "cpus_available": available_cpus(), "host_cpus": os.cpu_count(),
                       "workers": cfg["workers"], "python": platform.python_version(), "packages": packages,
                       "golden_run_s": float(golden[1]) if golden else None}

        phase("SCORE", scored=0, total=len(jobs))
        records = {}
        with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
            futures = {pool.submit(score, sha, name, onnx): sha for sha, (name, onnx) in jobs.items()}
            try:
                for count, future in enumerate(as_completed(futures), 1):
                    record = records[futures[future]] = future.result()
                    print(f"[{count}/{len(jobs)}] {record['policy_name']}: {record['score']:.3f}s, "
                          f"{record['seeds_finished']}/{len(record['per_seed'])} finished "
                          f"at {record['command_vx']:.2f} m/s", flush=True)
                    write_json(out / "status.json", {"phase": "SCORE", "state": "running",
                                                     "scored": count, "total": len(jobs)})
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

        phase("REPORT")

        def common(sha):
            record = records[sha]
            return {"common_score": record["score"], "common_seeds_finished": record["seeds_finished"],
                    "seeds_total": len(record["per_seed"]), "command_vx": record["command_vx"],
                    "run_id": record["run_id"], "model_sha256": sha}

        rows = [{"role": "candidate", **candidate, **common(candidate["model_sha256"])}
                for candidate in work_plan["candidates"]]
        rows.sort(key=lambda row: rank_key(row, "common_"))
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
            row["score_change_s"] = round(row["common_score"] - row["task_score"], 3)
        best = {}
        for row in rows:
            best.setdefault(row["forward_reward"], row)
        best_per_reward = [dict(best[reward], policy=f"arena/models/{best[reward]['model_sha256']}.onnx",
                                replay_route=f"/#/r/{best[reward]['run_id']}") for reward in sorted(best)]
        reference = None
        table = rows
        if work_plan["reference_sha"]:
            task_scores = {f"{task['settings']['forward_reward']:g}": task["reference"]["score"]
                           for task in work_plan["tasks"]}
            reference = {"role": "reference", **common(work_plan["reference_sha"]), "task_scores": task_scores}
            if len(set(task_scores.values())) == 1:
                reference["task_score"] = next(iter(task_scores.values()))
            table = [reference] + rows
        write_json(out / "comparison.json", table)
        with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(table)
        write_json(out / "best_per_reward.json", best_per_reward)
        run([sys.executable, "-m", "arena.cli", "board", "--event", EVENT, "--store", store],
            logs / "leaderboard.txt", env, cwd=work)

        winner = rows[0]
        summary = {"pipeline": "ok", "event": EVENT, "candidates": cfg["candidates"],
                   "conditions": work_plan["conditions"], "forward_rewards": sorted(best),
                   "candidates_scored": len(rows), "models_scored": len(jobs),
                   "meshive_task_id": os.environ.get("MESHIVE_SYS_TASK_ID") or os.environ.get("MESHIVE_TASK_ID"),
                   "winner": winner, "forward_reward": winner["forward_reward"],
                   "best_checkpoint": winner["checkpoint"], "mean_time_s": winner["common_score"],
                   "seeds_finished": winner["common_seeds_finished"], "seeds_total": winner["seeds_total"],
                   "command_vx": winner["command_vx"], "replay_run_id": winner["run_id"],
                   "policy": f"arena/models/{winner['model_sha256']}.onnx",
                   "has_full_finish": winner["common_seeds_finished"] == winner["seeds_total"],
                   "changed_from_task_scores": sum(row["score_change_s"] != 0 for row in rows),
                   "reference": reference,
                   "winner_beats_reference": None if reference is None else
                   (-winner["common_seeds_finished"], winner["common_score"])
                   < (-reference["common_seeds_finished"], reference["common_score"]),
                   "environment": environment}
        write_json(out / "summary.json", summary)
        (out / "README.txt").write_text(
            "Microduck Sprint common evaluation outputs\n\n"
            "summary.json: winner, reference and the evaluation environment (CPU, packages, golden run).\n"
            "comparison.csv/json: each candidate's Task score beside its common score, ranked.\n"
            "best_per_reward.json: the best common result for each forward reward, with replay routes.\n"
            "inputs.json: discovered Task outputs, shared training conditions and candidate count.\n"
            "arena/: fresh store holding only common-evaluation runs; import it into the viewer Pod.\n"
            "evaluation/: install, golden check, per-model score logs and the leaderboard.\n"
            "Ranking: most seeds finished first, then lowest mean time (falls/timeouts count 50 s).\n",
            encoding="utf-8")
        write_json(out / "status.json", {"phase": "DONE", "state": "complete", "pipeline_ok": True,
                                         "summary": summary})
        print("\n[MESHIVE] RESULT\n" + json.dumps(summary, indent=2), flush=True)
        if not summary["has_full_finish"]:
            print("No candidate completed all seeds; pipeline completed, skill goal not achieved.", flush=True)
    except BaseException as error:
        write_json(out / "status.json", {"phase": "FAILED", "state": "failed", "pipeline_ok": False,
                                         "error": str(error)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true", help="print the configuration and the input "
                        "plan (when the input root exists); install and score nothing")
    args = parser.parse_args(argv)
    cfg = settings(os.environ)
    if args.check_only:
        print(json.dumps(cfg, indent=2))
        if Path(cfg["input_root"]).is_dir():
            print(json.dumps(describe(plan(cfg)), indent=2))
        return
    execute(cfg, Path(os.environ.get("MESHIVE_OUTPUT_DIR") or "/outputs").resolve())


if __name__ == "__main__":
    main()
