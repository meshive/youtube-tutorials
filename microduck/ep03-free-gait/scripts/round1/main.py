"""Meshive Task: Microduck freestyle sprint. Train one style, score every checkpoint.

Built from Episode 1's task/main.py. Differences: the recipe is freestyle.py
(embedded below, written next to env.py), STYLE picks the variant, upstream's
tests are not run (they assert the walking knobs this recipe turns off), and
the export + Arena scoring path is rehearsed on the smoke checkpoint before
the long training starts.

Upload as main.py. Use Python 3.12. MODE defaults to smoke.
"""

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
STYLES = ("free", "hop", "glide")

FREESTYLE_SOURCE = r'''"""Freestyle sprint: train the race the Arena times, not a walking style.

Written into sprint_2m/ next to env.py by the Task script; env.py's make_env_cfg
is wrapped so that apply() below edits the upstream config. STYLE is filled in
by the Task script.

What every style shares:
  - no speed-command tracking and no walking-style rewards (posture, upright,
    foot lift, foot clearance, head pose); their weights are set to 0
  - reward is forward progress along the spawn heading, a cost per second, and
    a penalty for falling
  - a fall is the Arena's own rule: trunk tilted past 60 degrees for 0.3 s
  - an episode ends at the finish line (2.4 m: the Arena's 2 m plus room for
    the 0.5 s untimed settle) or after 5 s
  - no pushes and no standing practice; the other randomizations are upstream's

Styles:
  free   nothing else: whatever is fastest
  hop    feet must land and leave together, time in the air pays
  glide  feet must stay on the ground
"""

import torch
from mjlab.managers import ObservationTermCfg, RewardTermCfg, TerminationTermCfg

STYLE = "__STYLE__"
STYLES = ("free", "hop", "glide")

COURSE_M = 2.4
EPISODE_S = 5.0
FALL_GRAVITY_Z = -0.5  # arena/events/sprint-2m.toml
FALL_HOLD_STEPS = 15  # 0.3 s at 50 Hz

# Per second (the reward manager multiplies by the 0.02 s step).
PROGRESS_PER_M = 4.0
TIME_COST_PER_S = 4.0
FALL_PENALTY = 25.0
LATERAL_PER_M_S = 4.0

FEET_SENSOR = "feet_ground_contact"

DROPPED_REWARDS = (
    "track_linear_velocity",
    "track_angular_velocity",
    "upright",
    "pose",
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "head_pose_tracking",
    "head_pose_bias",
    "body_pose_tracking",
    "body_ang_vel",
    "angular_momentum",
)
DROPPED_CURRICULA = ("action_rate_weight", "head_pose_bias_weight", "standing_envs")


def _race(env):
    """Each env's start line (spawn position and heading) and fall counter."""
    robot = env.scene["robot"]
    pos = robot.data.root_link_pos_w[:, :2]
    w, x, y, z = robot.data.root_link_quat_w.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    state = getattr(env, "_freestyle", None)
    if state is None or state["pos"].shape[0] != env.num_envs:
        state = {
            "pos": pos.clone(),
            "yaw": yaw.clone(),
            "tipped": torch.zeros(env.num_envs, dtype=torch.long, device=pos.device),
            "fell": torch.zeros(env.num_envs, dtype=torch.bool, device=pos.device),
        }
        env._freestyle = state
    fresh = env.episode_length_buf <= 1
    state["pos"] = torch.where(fresh.unsqueeze(-1), pos, state["pos"])
    state["yaw"] = torch.where(fresh, yaw, state["yaw"])
    return robot, state, fresh


def _along_and_across(env):
    robot, state, _ = _race(env)
    delta = robot.data.root_link_pos_w[:, :2] - state["pos"]
    cos, sin = torch.cos(state["yaw"]), torch.sin(state["yaw"])
    return delta[:, 0] * cos + delta[:, 1] * sin, -delta[:, 0] * sin + delta[:, 1] * cos


def forward_velocity(env):
    """Trunk speed along the spawn heading, m/s. Not capped at the commanded speed."""
    robot, state, _ = _race(env)
    vel = robot.data.root_link_lin_vel_w[:, :2]
    speed = vel[:, 0] * torch.cos(state["yaw"]) + vel[:, 1] * torch.sin(state["yaw"])
    return torch.clamp(speed, -2.0, 4.0)


def lateral_offset(env):
    """Distance from the lane's centre line, m. The lane is 0.8 m wide."""
    return _along_and_across(env)[1].abs()


def time_cost(env):
    return torch.ones(env.num_envs, device=env.device)


def fell(env):
    """Termination: the Arena's fall, tilted past 60 degrees for 0.3 s straight."""
    robot, state, fresh = _race(env)
    tipped_now = robot.data.projected_gravity_b[:, 2] > FALL_GRAVITY_Z
    previous = torch.where(fresh, torch.zeros_like(state["tipped"]), state["tipped"])
    state["tipped"] = torch.where(tipped_now, previous + 1, torch.zeros_like(previous))
    state["fell"] = state["tipped"] >= FALL_HOLD_STEPS
    return state["fell"]


def fell_penalty(env):
    """1 on the step the fall termination fired (terminations run before rewards)."""
    _, state, _ = _race(env)
    return state["fell"].float()


def finished(env):
    """Termination: over the line."""
    return _along_and_across(env)[0] >= COURSE_M


def progress_obs(env):
    """Critic only: distance covered and offset from the centre line."""
    along, across = _along_and_across(env)
    return torch.stack([along, across], dim=-1)


def _feet_down(env):
    found = env.scene[FEET_SENSOR].data.found
    return found.reshape(env.num_envs, -1)[:, :2] > 0


def feet_out_of_sync(env):
    """1 while exactly one foot is on the ground."""
    down = _feet_down(env)
    return (down[:, 0] != down[:, 1]).float()


def flight(env):
    """1 while both feet are off the ground."""
    return (~_feet_down(env)).all(dim=-1).float()


def feet_in_air(env):
    """How many feet are off the ground: 0, 1 or 2."""
    return (~_feet_down(env)).float().sum(dim=-1)


def apply(cfg, play=False):
    if STYLE not in STYLES:
        raise ValueError(f"STYLE must be one of {STYLES}, not {STYLE!r}")

    for name in DROPPED_REWARDS:
        cfg.rewards[name].weight = 0.0
    for name in DROPPED_CURRICULA:
        del cfg.curriculum[name]
    cfg.rewards["action_rate_l2"].weight = -0.1  # upstream's starting value, held

    cfg.rewards["forward_velocity"] = RewardTermCfg(func=forward_velocity, weight=PROGRESS_PER_M)
    cfg.rewards["time_cost"] = RewardTermCfg(func=time_cost, weight=-TIME_COST_PER_S)
    cfg.rewards["lateral_offset"] = RewardTermCfg(func=lateral_offset, weight=-LATERAL_PER_M_S)
    # Weights are per second, so a one-step penalty is divided by the step.
    cfg.rewards["fell"] = RewardTermCfg(func=fell_penalty, weight=-FALL_PENALTY / 0.02)

    if STYLE == "hop":
        cfg.rewards["feet_out_of_sync"] = RewardTermCfg(func=feet_out_of_sync, weight=-4.0)
        cfg.rewards["flight"] = RewardTermCfg(func=flight, weight=1.5)
    elif STYLE == "glide":
        cfg.rewards["feet_in_air"] = RewardTermCfg(func=feet_in_air, weight=-4.0)

    cfg.terminations["fell_over"] = TerminationTermCfg(func=fell, time_out=False)
    cfg.terminations["finished"] = TerminationTermCfg(func=finished, time_out=False)
    cfg.episode_length_s = EPISODE_S

    cfg.events.pop("push_robot", None)
    cfg.commands["twist"].rel_standing_envs = 0.0
    cfg.observations["critic"].terms["progress"] = ObservationTermCfg(func=progress_obs)
    return cfg
'''

ENV_WRAPPER = '''

# --- freestyle: the upstream config above, edited by freestyle.apply ---
_upstream_make_env_cfg = make_env_cfg


def make_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    from . import freestyle

    return freestyle.apply(_upstream_make_env_cfg(play), play)
'''


def settings(environ):
    mode = environ.get("MODE", "smoke")
    if mode not in {"smoke", "train"}:
        raise ValueError("MODE must be smoke or train")
    cfg = {
        "mode": mode,
        "style": environ.get("STYLE", "free"),
        "num_envs": int(environ.get("NUM_ENVS", "64" if mode == "smoke" else "4096")),
        "max_iterations": int(environ.get("MAX_ITERATIONS", "2" if mode == "smoke" else "3000")),
        "seed": int(environ.get("SEED", "1")),
        "task": TASK,
        "event": EVENT,
        "repository": REPO_URL,
        "base_revision": BASE_REVISION,
    }
    if cfg["style"] not in STYLES:
        raise ValueError(f"STYLE must be one of {STYLES}")
    if not 1 <= cfg["num_envs"] <= 4096 or not 1 <= cfg["max_iterations"] <= 3000:
        raise ValueError("This recipe supports 1..4096 envs and 1..3000 iterations")
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
    work = Path(tempfile.mkdtemp(prefix="meshive-freestyle-"))
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

        folder = repo / "src/microduck_challenges/sprint_2m"
        recipe = folder / "challenge.toml"
        text = recipe.read_text()
        for section, key, value in [
            ("recipe", "num_envs", cfg["num_envs"]),
            ("recipe", "max_iterations", cfg["max_iterations"]),
            ("recipe", "seed", cfg["seed"]),
        ]:
            text = patch_value(text, section, key, value)
        tomllib.loads(text)
        recipe.write_text(text)
        if "__STYLE__" not in FREESTYLE_SOURCE:
            raise RuntimeError("freestyle.py has no STYLE placeholder")
        (folder / "freestyle.py").write_text(FREESTYLE_SOURCE.replace("__STYLE__", cfg["style"]), encoding="utf-8")
        env_file = folder / "env.py"
        env_file.write_text(env_file.read_text() + ENV_WRAPPER, encoding="utf-8")
        source = out / "source"
        source.mkdir()
        for filename in ["challenge.toml", "env.py", "freestyle.py"]:
            shutil.copy2(folder / filename, source / filename)
        shutil.copy2(repo / "uv.lock", source / "uv.lock")
        run(["git", "add", "src/microduck_challenges/sprint_2m"], logs / "git-add.log", repo)
        run(["git", "-c", "user.name=Meshive tutorial task", "-c", "user.email=tutorial@localhost",
             "commit", "-m", f"Freestyle recipe: style {cfg['style']}, seed {cfg['seed']}"],
            logs / "recipe-commit.log", repo)
        run([uv, "sync", "--frozen", "--group", "workshop"], logs / "install.log", repo)
        run(uv_run("python", "-c", "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
                   "print('torch', torch.__version__, 'GPU', torch.cuda.get_device_name(0))"),
            logs / "cuda-check.log", repo)
        run(uv_run("python", "-c",
                   "from microduck_challenges.sprint_2m import freestyle\n"
                   "from microduck_challenges.sprint_2m.env import make_env_cfg\n"
                   "cfg = make_env_cfg()\n"
                   "print('style', freestyle.STYLE, 'episode_s', cfg.episode_length_s)\n"
                   "print('rewards', {k: v.weight for k, v in cfg.rewards.items() if v.weight})\n"
                   "print('terminations', list(cfg.terminations))\n"
                   "print('curriculum', list(cfg.curriculum))\n"
                   "print('events', list(cfg.events))\n"),
            logs / "recipe-check.log", repo)

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

        store = out / "arena"
        records = store / "runs" / EVENT

        def export_and_score(run_dir, iteration, name, target_store):
            run(uv_run("publish", "--run", run_dir, "--checkpoint", iteration,
                       "--repo", f"local/{name}", "--name", name, "--dry-run"),
                logs / f"export-{name}.log", repo)
            exported = repo / f"publish-{name}"
            target_records = target_store / "runs" / EVENT
            before = set(target_records.glob("*.json"))
            run(uv_run("arena", "score", exported / "policy.onnx", "--event", EVENT,
                       "--user", "meshive", "--name", name, "--store", target_store),
                logs / f"score-{name}.log", repo)
            added = set(target_records.glob("*.json")) - before
            if len(added) != 1:
                raise RuntimeError("Evaluator did not save exactly one run record")
            return exported, json.loads(added.pop().read_text())

        def checkpoints_of(run_dir):
            return sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))

        phase("SMOKE")
        smoke_dir = train("task_smoke", 64, 2, logs / "smoke.log")
        smoke_checkpoints = checkpoints_of(smoke_dir)
        if not smoke_checkpoints:
            raise RuntimeError("No smoke checkpoint saved")
        # Rehearse export and scoring now, in a throwaway store: a recipe that
        # cannot be exported must fail here, not after the long training.
        smoke_iteration = int(smoke_checkpoints[-1].stem.split("_")[1])
        _, smoke_row = export_and_score(smoke_dir, smoke_iteration, f"smoke-{cfg['style']}-{run_tag}",
                                        work / "smoke-arena")
        write_json(out / "smoke.json", smoke_row)
        print(f"[MESHIVE] smoke rehearsal scored {smoke_row['score']} s, "
              f"{smoke_row['seeds_finished']} finished (an untrained policy is not expected to finish)", flush=True)

        if cfg["mode"] == "smoke":
            run_dir, checkpoints = smoke_dir, smoke_checkpoints[-1:]
        else:
            phase("TRAIN")
            run_dir = train(f"task_{cfg['style']}", cfg["num_envs"], cfg["max_iterations"], logs / "train.log")
            checkpoints = checkpoints_of(run_dir)
            if not checkpoints:
                raise RuntimeError("No checkpoints saved")

        phase("EVALUATE")
        results = []
        exports = {}
        for checkpoint in checkpoints:
            iteration = int(checkpoint.stem.split("_")[1])
            name = f"freestyle-{cfg['style']}-s{cfg['seed']}-{run_tag}-{iteration}"
            exported, row = export_and_score(run_dir, iteration, name, store)
            exports[iteration] = exported
            row["checkpoint"] = iteration
            row["style"] = cfg["style"]
            row["seed"] = cfg["seed"]
            results.append(row)
            write_json(out / "results.json", results)
            print(f"[MESHIVE] checkpoint {iteration}: {row['score']} s, "
                  f"{row['seeds_finished']}/{len(row['per_seed'])} at {row['command_vx']} m/s", flush=True)

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
            writer = csv.DictWriter(handle, fieldnames=["style", "seed", "checkpoint", "score", "seeds_finished", "command_vx", "run_id"])
            writer.writeheader()
            writer.writerows({key: row[key] for key in writer.fieldnames} for row in results)
        summary = {"mode": cfg["mode"], "task": TASK, "base_revision": BASE_REVISION,
                   "style": cfg["style"], "seed": cfg["seed"],
                   "num_envs": cfg["num_envs"], "max_iterations": cfg["max_iterations"],
                   "meshive_task_id": os.environ.get("MESHIVE_SYS_TASK_ID") or os.environ.get("MESHIVE_TASK_ID"),
                   "checkpoints_evaluated": len(results), "best_checkpoint": best["checkpoint"],
                   "mean_time_s": best["score"], "seeds_finished": best["seeds_finished"],
                   "seeds_total": len(best["per_seed"]), "command_vx": best["command_vx"],
                   "replay_run_id": best["run_id"], "policy": "best/policy.onnx",
                   "checkpoint": f"best/model_{best['checkpoint']}.pt",
                   "policy_sha256": hashlib.sha256((best_dir / "policy.onnx").read_bytes()).hexdigest(),
                   "has_full_finish": best["seeds_finished"] == len(best["per_seed"]),
                   "episode_2_record_s": 1.846}
        write_json(out / "summary.json", summary)
        write_json(out / "status.json", {"phase": "DONE", "state": "complete", "summary": summary})
        print("\n[MESHIVE] RESULT\n" + json.dumps(summary, indent=2), flush=True)
        if not summary["has_full_finish"]:
            print("No evaluated model completed all seeds; pipeline completed, skill goal not achieved.", flush=True)
    except BaseException as error:
        write_json(out / "status.json", {"phase": "FAILED", "state": "failed", "error": str(error)})
        raise


def main():
    cfg = settings(os.environ)
    if "--check-only" in sys.argv:
        print(json.dumps(cfg, indent=2))
        return
    execute(cfg, Path(os.environ.get("MESHIVE_OUTPUT_DIR") or "/outputs").resolve())


if __name__ == "__main__":
    main()
