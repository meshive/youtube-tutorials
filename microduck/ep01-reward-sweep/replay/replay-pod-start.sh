# Paste this whole file into the Pod's Connect terminal.
cat > /tmp/replay.sh <<'REPLAY_EOF'
set -euo pipefail
WHEEL=microduck_arena-0.5.1-py3-none-any.whl
URL=https://github.com/pollen-robotics/microduck-arena/releases/download/v0.5.1/$WHEEL
SHA=783049640bea382af3dfa0075844c3282152ba1a587b48678c6c81f965855eaf
IN=/inputs
STORE=/tmp/arena-store
W=/tmp/arena-wheel
mkdir -p "$W" "$STORE"
echo "[EP1] INSTALL"
python -m pip download --no-deps --no-cache-dir -q -d "$W" "$URL"
echo "$SHA  $W/$WHEEL" | sha256sum -c -
printf "%s\n" fastapi==0.141.1 starlette==1.7.0 uvicorn==0.54.0 httpx==0.28.1 huggingface-hub==2.0.0 websockets==16.1.1 python-multipart==0.0.32 > "$W/constraints.txt"
python -m pip install --no-cache-dir --disable-pip-version-check -q -c "$W/constraints.txt" "$W/$WHEEL"
echo "[EP1] WAIT INPUTS"
last=-1
for i in $(seq 40); do
  n=$(find "$IN" -type f 2>/dev/null | wc -l || true)
  if [ "$n" = "$last" ] && [ -n "$(find "$IN" -type d -path "*/runs/sprint-2m" -print -quit 2>/dev/null)" ]; then break; fi
  last=$n
  sleep 15
done
echo "[EP1] COPY"
roots=$(find "$IN" -type d -path "*/runs/sprint-2m" 2>/dev/null || true)
printf "%s\n" "$roots" | while read -r d; do
  if [ -z "$d" ]; then continue; fi
  r=$(dirname "$(dirname "$d")")
  for k in models traces runs; do
    if [ -d "$r/$k" ]; then cp -R "$r/$k" "$STORE/"; fi
  done
done
if [ ! -d "$STORE/runs/sprint-2m" ]; then
  echo "[EP1] FAILED no runs under $IN"
  find "$IN" -maxdepth 3 2>/dev/null | sed -n 1,40p || true
  exit 1
fi
python -c "
import glob, json
for p in sorted(glob.glob(\"$STORE/runs/sprint-2m/*.json\")):
    r = json.load(open(p))
    print(\"[EP1] RUN\", r.get(\"run_id\"), r.get(\"score\"), r.get(\"seeds_finished\"), str(r.get(\"model_sha256\"))[:12], flush=True)
"
echo "[EP1] SERVE"
export PYTHONUNBUFFERED=1
exec arena serve --store "$STORE" --user meshive --host 0.0.0.0 --port 8000
REPLAY_EOF
nohup bash /tmp/replay.sh > /tmp/replay.log 2>&1 &
tail -f /tmp/replay.log
