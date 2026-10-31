# Microduck 2 m sprint reward sweep: run materials

Video: coming soon

Scripts and settings from an experiment that trains the same robot with the forward reward set to 1 through 5, then compares which value runs 2 m fastest. It runs on Meshive Tasks and Pods.

## The four scripts

| File | Where it runs | What it does |
|---|---|---|
| `task/main.py` | Task, GPU | Installs the environment, trains, evaluates checkpoints, picks the best model |
| `common-eval/main.py` | Task, CPU | Rescores the five training results in one environment and saves the ranking |
| `replay/replay-pod-start.sh` | Pod terminal | Installs Arena, copies the results, starts the replay server |
| `replay/replay-keep-six.sh` | Pod terminal | Cuts the replay list to the best run per reward plus the reference model |

`settings/reward-1.env` to `reward-5.env` hold the environment variables for the five Tasks. Only `FORWARD_REWARD` differs.

## Steps

### 1. Five training Tasks

| Field | Value |
|---|---|
| Script | `task/main.py` |
| Image | `python:3.12` |
| Hardware | 1 × RTX 5090 |
| Max run time | 4h |
| Environment | Contents of `settings/reward-N.env` |
| Task name | `microduck-sprint-rN-s1-i3000` |

When training finishes, `/outputs` is saved to Assets as an asset with the same name as the Task.

### 2. Common evaluation Task

Each Task evaluated on a different machine, so this step rescores everything in one environment.

| Field | Value |
|---|---|
| Script | `common-eval/main.py` |
| Image | `python:3.12` |
| Hardware | CPU only → Compute |
| Max run time | 1h |
| Environment | `EXPECTED_TASKS=5`, `CANDIDATES=all` |

Add the five training outputs under Input assets. In each asset's file picker, select these five items. Attaching a whole asset goes over the limit of 500 input files.

- `settings.json`
- `status.json`
- `results.json`
- `reference.json`
- the `arena/models` folder

If any asset is missing `reference.json`, the Task stops with `Inputs disagree on the reference model`.

### 3. Replay Pod

Arena replays are drawn by the viewer's browser, so no GPU is needed.

1. Create a `python:3.12` CPU Pod.
2. Under Endpoints, add port 8000 and turn on external access.
3. Under Assets, add the common evaluation output and select only the `arena` folder.
4. Paste the contents of `replay/replay-pod-start.sh` into the Connect terminal.
5. When the log shows `[EP1] SERVE`, open the external address on the Connect tab.

The leaderboard lists every checkpoint, so it has many rows. To shorten it, change `keep` in `replay/replay-keep-six.sh` to your own results and paste it into the terminal. The values are the first 8 characters of `model_sha256` from `best_per_reward.json` and `summary.json`.

The replay server has no login. Keep the address private and delete the Pod when you are done.

## Pinned versions

| Item | Value |
|---|---|
| Training code | [pollen-robotics/microduck-challenges](https://github.com/pollen-robotics/microduck-challenges) `9aaec6db450dd91c7aa669fc7156d81ce43de3c7` |
| Scorer | microduck-arena 0.5.1 |
| Python | 3.12 |

## Actual cost

| Item | Billed |
|---|---:|
| Five training Tasks | $4.34 |
| Common evaluation | $0.02 |

These are the amounts at the time of recording. Prices can change.

## Results

The common evaluation rescored 66 models in one environment. The scorer finds each model's fastest commanded speed, then times it over 5 seeds.

| Reward | Fastest checkpoint | 2 m mean | Finished |
|---|---:|---:|---:|
| 5 | 1000 | 2.446 s | 5/5 |
| 4 | 1000 | 2.976 s | 5/5 |
| 3 | 1000 | 3.407 s | 5/5 |
| 2 | 2500 | 4.993 s | 5/5 |
| 1 | 500 | 23.323 s | 5/5 |
| Reference model | - | 4.886 s | 5/5 |

In this setup, a larger reward ran faster. For all five rewards, a checkpoint partway through training beat the last one.

Each reward was trained once with seed 1. A different seed could change the order.

## License

The scripts and settings in this folder are under the MIT license in the repository root. The training code and the scorer are not in this repository. They are downloaded from their sources at run time, so check each project for its terms of use.
