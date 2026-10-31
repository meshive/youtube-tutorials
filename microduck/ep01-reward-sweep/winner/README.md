# Episode 1 winning model

The fastest robot from the Episode 1 reward sweep, ready to load into Microduck Arena.

| Field | Value |
|---|---|
| Event | `sprint-2m` |
| Forward reward | 5 |
| Checkpoint | 1000 |
| Mean time | 2.446 s |
| Seeds finished | 5 of 5 |
| Commanded speed | 0.95 m/s |
| Arena version | 0.5.1 |
| Arena run ID | `01M67XTCATN3KPZ9H7VKXA` |
| Model SHA-256 | `74cff9b0cb69f52a7d1c051df8ee4b56908c1ca5a91fad9348417af5e60e79e5` |

## Files

`arena/` follows the Arena store layout, so you can copy it into a store and open it with `arena serve --store`.

| File | Contents |
|---|---|
| `arena/models/<sha256>.onnx` | The trained policy |
| `arena/runs/sprint-2m/<run_id>.json` | The scored run |
| `arena/traces/<run_id>.npz` | The replay trace |

The time is the mean of 5 seeds from the common evaluation, where falls and timeouts count as 50 s. It is a simulation result, and a replay shows one run.

## Terms

This model was trained with [`../task/main.py`](../task/main.py). The training environment and the scorer are downloaded at run time from Pollen Robotics projects, so check those projects for their terms of use.
