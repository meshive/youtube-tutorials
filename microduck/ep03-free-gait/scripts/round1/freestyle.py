"""Freestyle sprint: train the race the Arena times, not a walking style.

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
