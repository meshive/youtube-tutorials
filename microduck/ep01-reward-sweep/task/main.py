"""Meshive Task: Microduck Sprint smoke/train, evaluate, and retain replay data.

Upload as main.py. Use Python 3.12. MODE defaults to smoke, not a long run.
No remote publication, Hub upload, or cloud resource creation is performed here.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

REPO_URL = "https://github.com/pollen-robotics/microduck-challenges.git"
BASE_REVISION = "9aaec6db450dd91c7aa669fc7156d81ce43de3c7"
TASK = "Mjlab-Sprint2m-MicroDuck"
EVENT = "sprint-2m"


def settings(environ):
    mode = environ.get("MODE", "smoke")
    if mode not in {"smoke", "train"}:
        raise ValueError("MODE must be smoke or train")
    cfg = {
        "mode": mode,
        "forward_reward": float(environ.get("FORWARD_REWARD", "4.0")),
        "num_envs": int(environ.get("NUM_ENVS", "64" if mode == "smoke" else "4096")),
        "max_iterations": int(environ.get("MAX_ITERATIONS", "2" if mode == "smoke" else "1500")),
        "seed": int(environ.get("SEED", "1")),
        "task": TASK,
        "event": EVENT,
        "repository": REPO_URL,
        "base_revision": BASE_REVISION,
    }
    if not 0.5 <= cfg["forward_reward"] <= 5.0:
        raise ValueError("FORWARD_REWARD must be between 0.5 and 5.0")
    if not 1 <= cfg["num_envs"] <= 4096 or not 1 <= cfg["max_iterations"] <= 3000:
        raise ValueError("This recipe supports 1..4096 envs and 1..3000 iterations")
    if mode == "smoke" and (cfg["num_envs"] != 64 or cfg["max_iterations"] != 2):
        raise ValueError("Smoke mode uses exactly 64 envs and 2 iterations")
    return cfg


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def patch_value(text, section, key, value):
    pattern = rf"(?ms)(^\[{re.escape(section)}\][^\[]*?^{re.escape(key)}\s*=\s*)[^\n]+"
    text, count = re.subn(pattern, lambda match: match[1] + str(value), text, count=1)
    if count != 1:
        raise ValueError(f"Missing TOML setting: {section}.{key}")
    return text


def choose_best(rows):
    return min(rows, key=lambda row: (-row["seeds_finished"], row["score"], row["checkpoint"]))


def execute(cfg, out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "settings.json").exists():
        raise RuntimeError("Use a fresh Task output directory; settings.json already exists")
    write_json(out / "settings.json", cfg)
    logs = out / "evaluation"
    logs.mkdir()
    work = Path(tempfile.mkdtemp(prefix="meshive-sprint-"))
    run_tag = hashlib.sha256(str(work).encode()).hexdigest()[:8]
    repo = work / "microduck-challenges"
    env = dict(os.environ, PYTHONUNBUFFERED="1", MUJOCO_GL="egl", WANDB_MODE="disabled",
               UV_HTTP_TIMEOUT="600", UV_LINK_MODE="hardlink", UV_CACHE_DIR=str(work / "uv-cache"))

    def phase(name):
        print(f"\n[MESHIVE] {name}", flush=True)
        write_json(out / "status.json", {"phase": name, "state": "running"})

    def run(command, logfile, cwd=None):
        print("$ " + " ".join(map(str, command)), flush=True)
        with logfile.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    handle.write(line)
                    handle.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if code:
            raise subprocess.CalledProcessError(code, command)

    def uv_run(*args):
        return [uv, "run", "--frozen", "--group", "workshop", *map(str, args)]

    try:
        if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
            raise RuntimeError("This Task requires Linux with Python 3.12")
        phase("SETUP")
        run(["nvidia-smi"], logs / "gpu.log")
        run(["apt-get", "update", "-qq"], logs / "apt-update.log")
        run(["apt-get", "install", "-y", "-qq", "git", "ca-certificates", "libegl1", "libgl1",
             "libglvnd0", "libgles2"], logs / "system-packages.log")
        run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "uv==0.13.5"], logs / "uv-install.log")
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError("uv was not found after installation")
        run(["git", "init", repo], logs / "git-init.log")
        run(["git", "remote", "add", "origin", REPO_URL], logs / "git-remote.log", repo)
        run(["git", "fetch", "--depth", "1", "origin", BASE_REVISION], logs / "git-fetch.log", repo)
        run(["git", "checkout", "--detach", "FETCH_HEAD"], logs / "git-checkout.log", repo)
        lock = tomllib.loads((repo / "uv.lock").read_text())
        arena = next(p for p in lock["package"] if p["name"] == "microduck-arena")
        if arena["version"] != "0.5.1":
            raise RuntimeError("Unexpected Arena version")

        recipe = repo / "src/microduck_challenges/sprint_2m/challenge.toml"
        text = recipe.read_text()
        for section, key, value in [
            ("params.forward_reward", "value", cfg["forward_reward"]),
            ("recipe", "num_envs", cfg["num_envs"]),
            ("recipe", "max_iterations", cfg["max_iterations"]),
            ("recipe", "seed", cfg["seed"]),
        ]:
            text = patch_value(text, section, key, value)
        tomllib.loads(text)
        recipe.write_text(text)
        source = out / "source"
        source.mkdir()
        for filename in ["challenge.toml", "env.py"]:
            shutil.copy2(recipe.parent / filename, source / filename)
        shutil.copy2(repo / "uv.lock", source / "uv.lock")
        run(["git", "add", "src/microduck_challenges/sprint_2m/challenge.toml"], logs / "git-add.log", repo)
        run(["git", "-c", "user.name=Meshive tutorial task", "-c", "user.email=tutorial@localhost",
             "commit", "--allow-empty", "-m", f"Task recipe: forward reward {cfg['forward_reward']}, seed {cfg['seed']}"],
            logs / "recipe-commit.log", repo)
        run([uv, "sync", "--frozen", "--group", "workshop"], logs / "install.log", repo)
        run(uv_run("python", "-c", "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
                   "print('torch', torch.__version__, 'GPU', torch.cuda.get_device_name(0))"),
            logs / "cuda-check.log", repo)
        run(uv_run("pytest", "tests"), logs / "tests.log", repo)

        # Training keeps its expected path, while actual saved files land in outputs.
        training = out / "training"
        training.mkdir()
        training_link = repo / "logs/rsl_rl/sprint_2m"
        training_link.parent.mkdir(parents=True, exist_ok=True)
        training_link.symlink_to(training, target_is_directory=True)

        def train(name, count, iterations, filename):
            run(uv_run("train", TASK, "--env.scene.num-envs", count,
                       "--agent.max-iterations", iterations, "--agent.seed", cfg["seed"],
                       "--agent.logger", "tensorboard", "--agent.run-name", name), filename, repo)
            found = list(training.glob(f"*_{name}"))
            if len(found) != 1:
                raise RuntimeError(f"Expected one {name} run; found {len(found)}")
            return found[0]

        phase("SMOKE")
        smoke_dir = train("task_smoke", 64, 2, logs / "smoke.log")
        if cfg["mode"] == "smoke":
            run_dir = smoke_dir
        else:
            phase("TRAIN")
            run_dir = train(f"task_forward{cfg['forward_reward']:g}", cfg["num_envs"], cfg["max_iterations"], logs / "train.log")
        checkpoints = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        if not checkpoints:
            raise RuntimeError("No checkpoints saved")
        if cfg["mode"] == "smoke":
            checkpoints = checkpoints[-1:]

        phase("EVALUATE")
        store = out / "arena"
        records = store / "runs" / EVENT
        reference = None
        if cfg["mode"] == "train":
            reference_dir = work / "reference"
            run(uv_run("hf", "download", "pollen-robotics/microduck-sprint-2m-reference", "policy.onnx",
                       "--local-dir", reference_dir), logs / "reference-download.log", repo)
            before = set(records.glob("*.json"))
            run(uv_run("arena", "score", reference_dir / "policy.onnx", "--event", EVENT,
                       "--user", "meshive", "--name", "reference-this-task", "--store", store),
                logs / "reference-score.log", repo)
            added = set(records.glob("*.json")) - before
            if len(added) != 1:
                raise RuntimeError("Reference evaluator did not save exactly one record")
            reference = json.loads(added.pop().read_text())
            write_json(out / "reference.json", reference)
        results = []
        exports = {}
        for checkpoint in checkpoints:
            iteration = int(checkpoint.stem.split("_")[1])
            name = f"task-{cfg['mode']}-forward{cfg['forward_reward']:g}-{run_tag}-{iteration}"
            run(uv_run("publish", "--run", run_dir, "--checkpoint", iteration,
                       "--repo", f"local/{name}", "--name", name, "--dry-run"),
                logs / f"export-{iteration}.log", repo)
            exported = repo / f"publish-{name}"
            exports[iteration] = exported
            before = set(records.glob("*.json"))
            run(uv_run("arena", "score", exported / "policy.onnx", "--event", EVENT,
                       "--user", "meshive", "--name", name, "--store", store),
                logs / f"score-{iteration}.log", repo)
            added = set(records.glob("*.json")) - before
            if len(added) != 1:
                raise RuntimeError("Evaluator did not save exactly one run record")
            row = json.loads(added.pop().read_text())
            row["checkpoint"] = iteration
            row["forward_reward"] = cfg["forward_reward"]
            row["seed"] = cfg["seed"]
            results.append(row)
            write_json(out / "results.json", results)

        best = choose_best(results)
        best_dir = out / "best"
        best_dir.mkdir()
        exported = exports[best["checkpoint"]]
        for filename in ["policy.onnx", "manifest.json", "README.md"]:
            shutil.copy2(exported / filename, best_dir / filename)
        shutil.copy2(exported / "checkpoint.pt", best_dir / f"model_{best['checkpoint']}.pt")
        write_json(best_dir / "evaluation.json", best)
        run(uv_run("arena", "board", "--event", EVENT, "--store", store), logs / "leaderboard.txt", repo)
        with (out / "results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["forward_reward", "seed", "checkpoint", "score", "seeds_finished", "command_vx", "run_id"])
            writer.writeheader()
            writer.writerows({key: row[key] for key in writer.fieldnames} for row in results)
        summary = {"mode": cfg["mode"], "task": TASK, "base_revision": BASE_REVISION,
                   "forward_reward": cfg["forward_reward"], "seed": cfg["seed"],
                   "num_envs": cfg["num_envs"], "max_iterations": cfg["max_iterations"],
                   "meshive_task_id": os.environ.get("MESHIVE_SYS_TASK_ID") or os.environ.get("MESHIVE_TASK_ID"),
                   "checkpoints_evaluated": len(results), "best_checkpoint": best["checkpoint"],
                   "mean_time_s": best["score"], "seeds_finished": best["seeds_finished"],
                   "seeds_total": len(best["per_seed"]), "command_vx": best["command_vx"],
                   "replay_run_id": best["run_id"], "policy": "best/policy.onnx",
                   "checkpoint": f"best/model_{best['checkpoint']}.pt",
                   "policy_sha256": hashlib.sha256((best_dir / "policy.onnx").read_bytes()).hexdigest(),
                   "has_full_finish": best["seeds_finished"] == len(best["per_seed"]),
                   "reference_mean_time_s": reference["score"] if reference else None,
                   "reference_run_id": reference["run_id"] if reference else None}
        write_json(out / "summary.json", summary)
        (out / "README.txt").write_text(
            "Microduck Sprint Task outputs\n\n"
            "summary.json: actual best result; results.json/csv: all evaluated checkpoints.\n"
            "best/: validated ONNX export and corresponding training checkpoint.\n"
            "training/: all checkpoints, TensorBoard events, provenance and params.\n"
            "arena/: complete local replay store; import this into a viewer Pod.\n"
            "Smoke mode tests the pipeline; a smoke policy is not expected to finish.\n"
            "These are fresh Task results, not the historical 2.727 s Pod result.\n", encoding="utf-8")
        write_json(out / "status.json", {"phase": "DONE", "state": "complete", "summary": summary})
        print("\n[MESHIVE] RESULT\n" + json.dumps(summary, indent=2), flush=True)
        if not summary["has_full_finish"]:
            print("No evaluated model completed all seeds; pipeline completed, skill goal not achieved.", flush=True)
    except BaseException as error:
        write_json(out / "status.json", {"phase": "FAILED", "state": "failed", "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    cfg = settings(os.environ)
    if args.check_only:
        print(json.dumps(cfg, indent=2))
        return
    execute(cfg, Path(os.environ.get("MESHIVE_OUTPUT_DIR") or "/outputs").resolve())


if __name__ == "__main__":
    main()
