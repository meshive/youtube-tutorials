# Episode 2: Can an AI beat my record?

Video: https://youtu.be/EfYrVpU4tj0

In Episode 1 I trained a Microduck to sprint 2 meters in 2.446 seconds. In Episode 2 an AI agent gets the same code and a $10 budget, and I don't write any code. The agent is Claude Code, connected to Meshive through MCP. It read the reward code, planned three training runs, measured them and set up the race.

## What is here

| Folder | Contents |
|---|---|
| [`winner/`](winner) | The fastest model (Run A, 1.846 s), ready to load into Microduck Arena |

The agent wrote its scripts during the session, so there is no script folder for this episode. Its starting point was the Episode 1 code in [`../ep01-reward-sweep`](../ep01-reward-sweep). To connect your own agent, see https://docs.meshive.ai/sdk-cli/mcp/.

## Results

Each run changes one setting from the Episode 1 winner. Scoring is the same as Episode 1 (Arena 0.5.1, mean of 5 runs).

| Run | Change | Mean time | Finished | Fastest checkpoint |
|---|---|---|---|---|
| A | Smoothing penalty held at -0.6 | 1.846 s | 5 of 5 | 2,750 |
| C | Looser posture rules at speed | 1.993 s | 5 of 5 | 1,250 |
| B | Upright reward 1.0 instead of 2.0 | 2.063 s | 5 of 5 | 2,250 |
| Episode 1 record | | 2.446 s | 5 of 5 | 1,000 |

Each run was trained once with seed 1, so another seed could change the numbers. These are simulation times.

## Actual cost

Total billed: $2.63 of the $10 budget. The three training Tasks cost $0.88 + $0.86 + $0.86 at $0.683/hr, one RTX 5090 each. Billed amounts at the time of recording. Prices can change.
