#!/usr/bin/env bash
# Local read-only dashboard for the 9-run campaign.
#
# Polls tools/dash_feed.py on the box every INTERVAL seconds, drops each frame
# into runs/_dash/data.json, and serves that directory over http://127.0.0.1 so
# a browser can auto-refresh. Nothing is installed on the box and nothing is
# written to it: dash_feed.py only reads metrics.jsonl, nvidia-smi and the logs.
#
#   TC_PASS=... bash tools/live_dashboard.sh          # password from the env
#   bash tools/live_dashboard.sh                      # prompts (needs a tty)
#   PORT=8766 INTERVAL=30 bash tools/live_dashboard.sh
#
# The password is handed to ssh through a temporary SSH_ASKPASS script inside
# runs/_dash/ (mode 600, removed on exit). It is never written to the shell
# history, the repo, or any tracked file; runs/* is git-ignored.
set -uo pipefail
cd "$(dirname "$0")/.."

HOST="${HOST:-hz01-ssh.gpuhome.cc}"
PORT="${PORT:-30123}"
RUSER="${RUSER:-root}"
RDIR="${RDIR:-/root/rivermind-data/two-clocks-audit}"
RPY="${RPY:-/opt/conda/bin/python}"
INTERVAL="${INTERVAL:-15}"
HTTP_PORT="${HTTP_PORT:-8765}"
PY_WEB="${PY_WEB:-C:/Users/Administrator/.workbuddy/binaries/python/versions/3.13.12/python.exe}"

DASH="runs/_dash"
# NOT inside the repo: this checkout lives under a path with spaces
# ("work sapce 1"), and ssh cannot execute an SSH_ASKPASS whose path has one.
ASK="/tmp/tc_ask_$$.sh"
ERR="$DASH/last_error.txt"
mkdir -p "$DASH"

PW="${TC_PASS:-}"
if [[ -z "$PW" ]]; then
  if [[ -t 0 ]]; then
    read -rsp "SSH password for ${RUSER}@${HOST}: " PW; echo
  else
    echo "no tty and TC_PASS unset: cannot ask for the password" >&2
    exit 1
  fi
fi
[[ -z "$PW" ]] && { echo "empty password" >&2; exit 1; }

cleanup() {
  rm -f "$ASK"
  [[ -n "${WEBPID:-}" ]] && kill "$WEBPID" 2>/dev/null
  echo "[dash] stopped; askpass removed"
}
trap cleanup EXIT INT TERM

# ssh needs an executable helper to answer the prompt; there is no way to hand
# it a password any other way, so this file exists for the life of the process.
printf '#!/bin/sh\necho %q\n' "$PW" >"$ASK"
chmod 700 "$ASK" 2>/dev/null || true

cp -f tools/dashboard.html "$DASH/index.html"

if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:${HTTP_PORT}/" 2>/dev/null; then
  echo "[dash] port ${HTTP_PORT} already serving, reusing it"
else
  "$PY_WEB" -m http.server "$HTTP_PORT" --bind 127.0.0.1 --directory "$DASH" \
      >"$DASH/http.log" 2>&1 &
  WEBPID=$!
  sleep 1
fi
echo "[dash] http://127.0.0.1:${HTTP_PORT}/  (poll ${INTERVAL}s, target ${RDIR})"

SSHOPT=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
        -o ConnectTimeout=20 -o ServerAliveInterval=15
        -o NumberOfPasswordPrompts=1 -o PreferredAuthentications=password
        -o PubkeyAuthentication=no)

fails=0
while :; do
  out=$(SSH_ASKPASS="$ASK" SSH_ASKPASS_REQUIRE=force DISPLAY=:0 \
        ssh "${SSHOPT[@]}" -p "$PORT" "${RUSER}@${HOST}" \
        "cd '$RDIR' && '$RPY' tools/dash_feed.py" 2>>"$ERR")
  # A frame must be one JSON object; anything else (ssh banner, traceback,
  # empty output from a dropped connection) leaves the previous frame in place
  # and bumps the failure counter, which the UI shows as "stale".
  if [[ "$out" == \{*\} ]]; then
    printf '%s\n' "$out" >"$DASH/data.tmp" && mv -f "$DASH/data.tmp" "$DASH/data.json"
    : >"$ERR"
    fails=0
    free_mib=$(printf '%s' "$out" | sed -n 's/.*"free": *\([0-9]*\).*/\1/p' | head -1)
    printf '%s frame ok free=%s MiB\n' "$(date -u +%FT%TZ)" "${free_mib:-?}"
  else
    fails=$((fails+1))
    printf '%s poll failed (%d in a row)\n' "$(date -u +%FT%TZ)" "$fails"
    sleep 5
  fi
  sleep "$INTERVAL"
done
