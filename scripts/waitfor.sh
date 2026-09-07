#!/usr/bin/env bash
# Wait for a background job, exiting on EITHER outcome — never only on success.
#
# PREREGISTRATION.md 3e, instance 4: a loop polling for a success string with no
# failure condition showed a dead job as Running for 2.5 hours. A wait must be able
# to distinguish "still working" from "died", or it is not a wait, it is a hang.
#
#   scripts/waitfor.sh <pid> <logfile> <success-regex> [timeout-seconds]
#
# Exit: 0 success string seen | 1 process died without it | 2 timed out
set -uo pipefail
PID="${1:?pid}"; LOG="${2:?logfile}"; OK="${3:?success regex}"; TIMEOUT="${4:-86400}"
START=$(date +%s)
while true; do
  if grep -qE "$OK" "$LOG" 2>/dev/null; then echo "WAIT: success"; exit 0; fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "WAIT: FAILED — pid $PID is gone and '$OK' never appeared. Last lines:" >&2
    tail -5 "$LOG" >&2; exit 1
  fi
  if (( $(date +%s) - START > TIMEOUT )); then
    echo "WAIT: TIMEOUT after ${TIMEOUT}s; pid $PID still alive" >&2; exit 2
  fi
  sleep 10
done
