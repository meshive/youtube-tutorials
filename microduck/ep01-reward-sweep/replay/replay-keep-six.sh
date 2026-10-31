# Paste into the Pod's Connect terminal. Reduces the replay store to the best run per reward plus the reference model.
# The originals under /inputs are not touched. All other records move to /tmp/arena-hidden.
# The keep values are the results from this video. For your own run, use the first 8 characters of model_sha256 from best_per_reward.json and summary.json.
python - <<'KEEP_EOF'
import glob, json, os, shutil
runs = "/tmp/arena-store/runs/sprint-2m"
hidden = "/tmp/arena-hidden"
keep = {"f9019a85": "reward 1", "07ed5a2c": "reward 2", "45219592": "reward 3",
        "c26bba1a": "reward 4", "74cff9b0": "reward 5", "adbc1594": "reference"}
os.makedirs(hidden, exist_ok=True)
found = {}
for path in sorted(glob.glob(runs + "/*.json")):
    record = json.load(open(path))
    head = str(record.get("model_sha256"))[:8]
    if head in keep and head not in found:
        found[head] = (record.get("run_id"), record.get("score"), record.get("seeds_finished"))
    else:
        shutil.move(path, hidden)
for head, label in keep.items():
    print(label, found.get(head, "MISSING"))
print("kept", len(glob.glob(runs + "/*.json")), "moved", len(os.listdir(hidden)))
KEEP_EOF
