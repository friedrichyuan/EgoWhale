#!/usr/bin/env bash
# Show where a running batch is. Run it in a second shell on the same node while batch or bench is running.
# Nothing is printed by batch until every actor has loaded its model, and actor output goes to Ray's
# worker logs, not the batch log. This prints what each actor process is doing, the tail of the logs of
# those still busy, recent tracebacks, and GPU memory.
# Usage: scripts/diagnose.sh
set -uo pipefail
logs=${RAY_TMPDIR:-/tmp/ray}/session_latest/logs

echo "== driver"
pgrep -af "egowhale.batch" | grep -v pgrep || echo "no egowhale.batch process"

echo
echo "== actor processes: count, title, longest seconds in this state"
ps -eo etimes=,args= | awk '$2 ~ /^ray::/ {n[$2]++; if ($1 > t[$2]) t[$2] = $1} END {for (k in n) printf "%5d  %-36s %6ds\n", n[k], k, t[k]}' | sort -k2
echo "A title ending in .__init__ is still loading. No ray:: lines at all means Ray itself has not started."

echo
echo "== log tails of actors still in __init__"
ps -eo pid=,args= | awk '$2 ~ /^ray::.*\.__init__$/ {print $1, $2}' | while read -r pid title; do
    echo "-- $title pid $pid"
    for file in "$logs"/worker-*-"$pid".out "$logs"/worker-*-"$pid".err; do
        [ -s "$file" ] && tail -n 5 "$file"
    done
    if command -v py-spy > /dev/null; then
        py-spy dump --pid "$pid" 2> /dev/null | head -n 25
    fi
done

echo
echo "== latest tracebacks in worker logs"
found=0
for file in $(ls -t "$logs"/worker-*.err 2> /dev/null); do
    if grep -q "Traceback" "$file"; then
        echo "-- $file"
        grep -A 30 "Traceback" "$file" | tail -n 20
        found=$((found + 1))
        [ $found -ge 3 ] && break
    fi
done
[ $found -eq 0 ] && echo "none"

echo
echo "== GPUs"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
