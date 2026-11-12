# Episode 2 winning model

The fastest robot from Episode 2, where an AI agent ran the experiments through the Meshive MCP server. Ready to load into Microduck Arena.

| Field | Value |
|---|---|
| Event | `sprint-2m` |
| Forward reward | 5 |
| Change from the Episode 1 recipe | The action-rate (smoothness) penalty stops at -0.6 instead of ramping to -1.0 |
| Training seed | 1 |
| Iterations | 3000, 4096 environments |
| Checkpoint | 2750 |
| Mean time | 1.846 s |
| Seeds finished | 5 of 5 |
| Commanded speed | 1.20 m/s |
| Arena version | 0.5.1 |
| Arena run ID | `01M6JEZ6T6NW6QCB0QYKP4` |
| Model SHA-256 | `46af40fa24b0e342da021e299212801a0cf42b1db05b4792faf9f2481378a999` |

## What changed

Training used `pollen-robotics/microduck-challenges` at revision `9aaec6db450dd91c7aa669fc7156d81ce43de3c7` with two edits:

1. `challenge.toml`: the forward reward is 5.0 instead of 2.0. This is the Episode 1 winning setting.
2. `env.py`: the action-rate penalty curriculum ends at -0.6. Upstream sets -0.8 at iteration 1250 and -1.0 at iteration 1500, and this run sets -0.6 at both.

Everything else is upstream: upright reward 2.0, step reward 3.0, and the posture rewards unchanged. One training seed was run, so the result has not been repeated.

## Files

`arena/` follows the Arena store layout, so you can copy it into a store and open it with `arena serve --store`.

| File | Contents |
|---|---|
| `arena/models/<sha256>.onnx` | The trained policy |
| `arena/runs/sprint-2m/<run_id>.json` | The scored run |
| `arena/traces/<run_id>.npz` | The replay trace |

The time is the mean of 5 evaluation seeds, where falls and timeouts count as 50 s. It is a simulation result, and a replay shows one run.

## Terms

The training environment and the scorer are downloaded at run time from Pollen Robotics projects, so check those projects for their terms of use.
