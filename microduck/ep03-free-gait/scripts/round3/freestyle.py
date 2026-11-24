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
  - an episode ends at a line 3.2 m from the start or after 5 s. Round 1 ended
    at 2.4 m and its fast checkpoints fell 1.3 to 2.0 m into the Arena's timed
    2 m, which starts after a 0.5 s untimed run-up: they had learned a sprint
    that only lasts as far as the training line
  - no pushes and no standing practice; the other randomizations are upstream's

Styles scale the progress reward instead of adding penalties, so standing
still never pays (round 1's hop and glide stood still to avoid their penalties):
  free   progress always counts in full
  hop    full while both feet are down, 1.5x while both are in the air,
         0.2x while only one foot is down
  glide  full while both feet are down, 0.3x with one foot up, nothing in the air

hop and glide also have a pace rule: after 1.5 s the robot must have covered
0.2 m per second since, or it counts as a fall. Standing still is then never
the safe choice. free keeps round 1's rules, which did learn to move.
"""

import torch
from mjlab.managers import ObservationTermCfg, RewardTermCfg, TerminationTermCfg

STYLE = "__STYLE__"
STYLES = ("free", "hop", "glide")

COURSE_M = 3.2
EPISODE_S = 5.0
FALL_GRAVITY_Z = -0.5  # arena/events/sprint-2m.toml
FALL_HOLD_STEPS = 15  # 0.3 s at 50 Hz

# Per second (the reward manager multiplies by the 0.02 s step).
PROGRESS_PER_M = 4.0
TIME_COST_PER_S = 4.0
FALL_PENALTY = 25.0
LATERAL_PER_M_S = 4.0
PACE_M_PER_S = 0.2
PACE_GRACE_S = 1.5

# Progress multiplier by number of feet on the ground: (none, one, both).
STYLE_SCALE = {
    "free": (1.0, 1.0, 1.0),
    "hop": (1.5, 0.2, 1.0),
    "glide": (0.0, 0.3, 1.0),
}

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
    speed = torch.clamp(speed, -2.0, 4.0)
    if STYLE == "free":
        return speed
    scale = torch.tensor(STYLE_SCALE[STYLE], device=speed.device)[_feet_down(env).long().sum(dim=-1)]
    # Only forward motion is scaled: going backwards costs the same in every style.
    return torch.where(speed > 0, speed * scale, speed)


def lateral_offset(env):
    """Distance from the lane's centre line, m. The lane is 0.8 m wide."""
    return _along_and_across(env)[1].abs()


def time_cost(env):
    return torch.ones(env.num_envs, device=env.device)


def fell(env):
    """Termination: the Arena's fall (tilted past 60 degrees for 0.3 s straight); for hop and glide, also behind pace."""
    robot, state, fresh = _race(env)
    tipped_now = robot.data.projected_gravity_b[:, 2] > FALL_GRAVITY_Z
    previous = torch.where(fresh, torch.zeros_like(state["tipped"]), state["tipped"])
    state["tipped"] = torch.where(tipped_now, previous + 1, torch.zeros_like(previous))
    state["fell"] = state["tipped"] >= FALL_HOLD_STEPS
    if STYLE != "free":
        required = PACE_M_PER_S * (env.episode_length_buf * 0.02 - PACE_GRACE_S)
        state["fell"] = state["fell"] | (_along_and_across(env)[0] < required)
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

    cfg.terminations["fell_over"] = TerminationTermCfg(func=fell, time_out=False)
    cfg.terminations["finished"] = TerminationTermCfg(func=finished, time_out=False)
    cfg.episode_length_s = EPISODE_S

    cfg.events.pop("push_robot", None)
    cfg.commands["twist"].rel_standing_envs = 0.0
    cfg.observations["critic"].terms["progress"] = ObservationTermCfg(func=progress_obs)
    return cfg
