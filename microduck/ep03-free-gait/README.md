# Episode 3: No rules on how to move

Video: [link added at upload]

In Episode 2 an AI agent trained a Microduck to sprint 2 meters in 1.846 seconds, with a rule to keep a real walking gait. In Episode 3 the same agent (Claude Code, connected to Meshive through MCP) gets the same race and scorer, a $10 budget, and no rule about how to move. It read the scorer's code, wrote three recipes, ran two rounds of training and replayed every Microduck in the Arena.

The fastest Microduck finished in 1.27 to 1.35 seconds in four races out of five, and fell in the fifth. A fall counts as 50 seconds, so its mean is 11.047 seconds, and Episode 2's walking Microduck keeps the record. **Can you make the fast one stay up?** Everything we trained is here.

## What is here

| Folder | Contents |
|---|---|
| [`scripts/round1/`](scripts/round1) | The Task script (`main.py`) and recipe (`freestyle.py`) of round 1: free, hop, glide |
| [`scripts/round2-stopped/`](scripts/round2-stopped) | Round 2 as first written (lane training). Both runs stood still and the agent stopped them after about 40 minutes |
| [`scripts/round3/`](scripts/round3) | Round 3: longer training course (3.2 m), pace rule for hop and glide |
| [`scripts/seed2/`](scripts/seed2) | Episode 2's winning recipe run again with seed 2 |
| [`runs/`](runs) | For each training: every scored checkpoint as an Arena model (`.onnx`), its result (`runs/sprint-2m/*.json`) and its replay (`traces/*.npz`) |

All scripts were written by the agent during the session. They start from the Episode 1 code in [`../ep01-reward-sweep`](../ep01-reward-sweep) and clone [pollen-robotics/microduck-challenges](https://github.com/pollen-robotics/microduck-challenges). Upload `main.py` as a Meshive Task (Python 3.12, one RTX 5090); `MODE` defaults to a short smoke run.

## Results

Arena 0.5.1, event `sprint-2m`: mean of 5 races, a fall (trunk tilted past 60 degrees for 0.3 s) counts as 50 s. A Microduck that falls while crossing the line still finishes. Every run below used seed 1 unless noted.

| Training | Best checkpoint | Mean time | Finished | Note |
|---|---|---|---|---|
| Round 1, free | 1,000 | 2.588 s | 5 of 5 | Checkpoint 2,999: 1.32, 1.35, 1.29, 1.27 s and one fall, mean 11.047 s |
| Round 1, hop | — | 50 s | 0 of 5 | Falls at the start |
| Round 1, glide | — | 50 s | 0 of 5 | Leans back and never leaves the start |
| Round 3, free | 1,750 | 2.879 s | 5 of 5 | Checkpoint 2,999 still fell 1.37 to 1.92 m into the race, like round 1 (1.34 to 1.99 m) |
| Round 3, glide | 250 | 24.000 s | 3 of 5 | Finished races about 6.7 s |
| Round 3, hop | 1,000 | 45.317 s | 1 of 5 | Crawled 26.6 s to the line |
| Episode 2 recipe, seed 2 | 2,999 | 1.930 s | 5 of 5 | Episode 2's seed 1 was 1.846 s |

These are simulation times.

## Load a model into Microduck Arena

Each `runs/<training>/arena` folder is an Arena store (`models/`, `runs/sprint-2m/`, `traces/`). Copy the files of the checkpoint you want into your own store, or point `arena serve --store` at the folder to replay them all.

## Actual cost

Total billed: $7.92 of the $10 budget, in the workspace used for this episode: nine training Tasks $7.58 (one RTX 5090 each) and the replay Pod $0.34. Billed amounts at the time of recording. Prices can change.
