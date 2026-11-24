#!/usr/bin/env bash
# mem-floor-valve.sh — keep a node's margin above its reserve by dropping clean
# file cache once it has built up. "The host memory floor" in the README
# describes the state this tool is for.
#
# THE STATE IT FIXES
# The kernel lets page cache grow until free memory reaches its watermark, and
# reclaims it only when an allocation presses against that line. A busy node
# therefore settles just above its reserve while still holding clean file
# cache it does not need: each request that needs memory pays a reclaim
# under pressure, and there is no cushion for a burst. The node that runs the
# API server gets there first — it carries the API server and the engine core
# on top of its own model rank, about 1.5–2 GiB more host memory than the
# others.
#
# HOW IT DECIDES
# The margin is true headroom: (Normal zone `pages free` − `min`) × 4 KiB —
# what an ordinary allocation can still take before the kernel must reclaim.
# Under FLOOR_MIB, once the wait since the last drop is over, the valve asks
# the kernel to drop clean cache and reads the margin again.
#
# It does not try to predict what a drop will recover. What a drop can free
# is clean file pages that no process has mapped, and the kernel does not
# report that split: `Active(file)` + `Inactive(file)` counts every clean
# file page, mapped or not; `Mapped` counts mapped file pages and mapped
# shared memory together; and shared memory is swap-backed, sits on the
# anonymous lists, and is in neither of the file-page figures. So the valve
# measures instead. After a drop that gives back less than
# MIN_DROPPABLE_MIB it doubles the wait before the next one, up to MAX_GAP.
# A drop that recovers little is thus repeated at most once per MAX_GAP, and
# one that recovers a useful amount is followed by the next as soon as
# MIN_GAP later.
#
# WHAT IT DOES NOT DO
# It never stops, starts, signals or inspects the engine's container; its only
# privileged act is one call to a helper that drops clean cache. It never
# drops on a margin it could not read. It cannot create memory: when the
# margin is low because of what the processes themselves hold, a drop
# recovers little, and the valve says so in its log and waits longer. And it
# does nothing it cannot record: it stops rather than run without its log.
#
# Settings (environment variables; the defaults are the shipped values):
#   FLOOR_MIB=1000                          drop below this much true headroom
#   INTERVAL=10                             seconds between readings
#   MIN_GAP=30                              the wait after a drop that recovered
#                                           at least MIN_DROPPABLE_MIB
#   MAX_GAP=600                             the wait doubles after each drop that
#                                           recovered less, up to this
#   MIN_DROPPABLE_MIB=128                   what a drop must recover to count
#   RECLAIM=/usr/local/sbin/vm-drop-caches  the helper (tools/vm-drop-caches)
#   STATE_DIR=${XDG_STATE_HOME:-$HOME/.local/state}/ds41
#                                           mem-floor.log and mem-floor.csv
#
# Usage:
#   mem-floor-valve.sh             run the loop (normally under systemd)
#   mem-floor-valve.sh --status    print one reading and its decision, then exit
#   VALVE_LIB_ONLY=1 (when sourced) define the functions without running
set -u -o pipefail

FLOOR_MIB="${FLOOR_MIB:-1000}"
INTERVAL="${INTERVAL:-10}"
MIN_GAP="${MIN_GAP:-30}"
MAX_GAP="${MAX_GAP:-600}"
MIN_DROPPABLE_MIB="${MIN_DROPPABLE_MIB:-128}"
RECLAIM="${RECLAIM:-/usr/local/sbin/vm-drop-caches}"
# XDG_STATE_HOME counts only when it is an absolute path.
case "${XDG_STATE_HOME:-}" in
  /*) default_state="$XDG_STATE_HOME/ds41" ;;
  *)  default_state="$HOME/.local/state/ds41" ;;
esac
STATE_DIR="${STATE_DIR:-$default_state}"
LOG="$STATE_DIR/mem-floor.log"
CSV="$STATE_DIR/mem-floor.csv"
CSV_HEADER='utc,headroom_mib,file_pages_mib,shmem_mib,mapped_mib,mem_available_mib,mem_free_mib,decision'
# Where the readings come from. The test points these at fixture files.
ZONEINFO="${VALVE_ZONEINFO:-/proc/zoneinfo}"
MEMINFO="${VALVE_MEMINFO:-/proc/meminfo}"

# ---- readings: each a pure function of one /proc file ------------------------
# Each takes an optional path (the test passes a fixture) and prints EMPTY,
# never 0, when it cannot read what it needs.
zone_headroom_mib() {   # (Normal-zone pages free − min) × 4 KiB; may be negative
  awk '/^Node 0, zone/ {if (z) exit; z = ($4 == "Normal"); next}
       z && /pages free/ && !/nr_free/ {f=$3}
       z && /^ +min +[0-9]/ && !/ratio/ {m=$2}
       END {if (f == "" || m == "") exit; printf "%d", (f-m)*4/1024}' "${1:-$ZONEINFO}" 2>/dev/null
}
cache_mib() {           # prints: clean file pages, shared memory, mapped (MiB, space-separated)
  awk -F: '$1=="Active(file)" {af=$2} $1=="Inactive(file)" {inact=$2} $1=="Shmem" {s=$2} $1=="Mapped" {m=$2}
       END {if (af == "" || inact == "" || s == "" || m == "") exit
            printf "%d %d %d\n", (af+inact)/1024, s/1024, m/1024}' "${1:-$MEMINFO}" 2>/dev/null
}
meminfo_mib() {         # one key of /proc/meminfo, in MiB
  awk -F: -v k="$1" '$1==k {printf "%d", $2/1024; f=1} END {if (!f) exit}' "${2:-$MEMINFO}" 2>/dev/null
}

# ---- the decision: pure functions (tools/test-mem-floor-valve.sh imports them)
# valve_decide $1 headroom_mib  $2 seconds since the last drop  $3 the streak
#   (consecutive drops that recovered less than MIN_DROPPABLE_MIB).
# Prints DROP, or WAIT:<reason>. Nothing unreadable ever drops.
valve_decide() {
  local hr="$1" since="$2" streak="$3" gap
  # An integer, possibly NEGATIVE: free below the reserve line is the moment
  # this valve matters most. Only a non-integer is unreadable.
  case "${hr#-}" in ''|*[!0-9]*) echo "WAIT:unreadable"; return ;; esac
  if [ "$hr" -ge "$FLOOR_MIB" ]; then echo "WAIT:headroom-ok"; return; fi
  # An unreadable clock waits. (An errored comparison reads as false, which
  # is the opposite of what a drop guard must do.)
  case "$since" in ''|*[!0-9]*) echo "WAIT:throttled"; return ;; esac
  case "$streak" in ''|*[!0-9]*) streak=0 ;; esac
  # A gap that is not a whole number waits: an errored comparison reads as
  # false, which is a DROP.
  gap="$(valve_gap "$streak")"
  case "$gap" in ''|*[!0-9]*) echo "WAIT:throttled"; return ;; esac
  if [ "$since" -lt "$gap" ]; then echo "WAIT:throttled"; return; fi
  echo "DROP"
}
valve_gap() {           # $1 the streak -> the wait before the next drop, in seconds
  # EMPTY if either gap is not a whole number: the caller then waits.
  local gap="$MIN_GAP" cap="$MAX_GAP" i=0
  case "$gap" in ''|*[!0-9]*) return ;; esac      # each on its own: checked as one
  case "$cap" in ''|*[!0-9]*) return ;; esac      # string, an empty one can hide
  # Base 10 explicitly: a zero-padded value such as 0090 would otherwise be
  # read as octal — an error at 8 or 9, a different number below that.
  gap=$((10#$gap)); cap=$((10#$cap))
  case "$1" in ''|*[!0-9]*) set -- 0 ;; esac      # an unreadable streak waits the minimum
  while [ "$i" -lt "$1" ] && [ "$gap" -lt "$cap" ]; do gap=$(( gap * 2 )); i=$(( i + 1 )); done
  if [ "$gap" -gt "$cap" ]; then gap="$cap"; fi
  printf '%s' "$gap"
}
valve_recovered() {     # $1 headroom before a drop  $2 after -> the difference; EMPTY if either is unreadable
  # Each side is checked on its own: checked as one string, an empty "before"
  # hides behind a readable "after".
  case "${1#-}" in ''|*[!0-9]*) return ;; esac
  case "${2#-}" in ''|*[!0-9]*) return ;; esac
  echo $(( $2 - $1 ))
}
valve_next_streak() {   # $1 MiB the last drop recovered (negative if a burst ate more than it freed)  $2 the streak before it
  local rec="$1" before="$2"
  case "${rec#-}" in ''|*[!0-9]*) rec=0 ;; esac            # unreadable counts as "recovered little"
  case "$before" in ''|*[!0-9]*) before=0 ;; esac
  if [ "$rec" -lt "$MIN_DROPPABLE_MIB" ]; then echo $(( before + 1 )); else echo 0; fi
}

# ---- the loop -----------------------------------------------------------------
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"; }
# A line the valve may not continue without. If it cannot be written, the
# valve stops: it must not drop anything it has not recorded.
must_log() {
  log "$@" || { printf 'mem-floor-valve: cannot write its log (%s) — stopping\n' "$LOG" >&2; exit 1; }
}

# The one privileged act. The helper is a fixed root-owned script, and sudoers
# grants exactly that path without a password (see the README).
valve_reclaim() {
  if [ "$(id -u)" -eq 0 ]; then "$RECLAIM"; else sudo -n "$RECLAIM"; fi
}

die_state() { printf 'mem-floor-valve: %s (STATE_DIR=%s)\n' "$1" "$STATE_DIR" >&2; exit 1; }

valve_loop() {
  # A valve that cannot record what it does must not do it. The checks are
  # real writes or real permission tests, not `touch`: `touch` succeeds on an
  # existing read-only file.
  mkdir -p "$STATE_DIR" 2>/dev/null || die_state "cannot create its state directory"
  [ -w "$STATE_DIR" ]   || die_state "its state directory is not writable"
  if [ ! -f "$CSV" ] || [ ! -s "$CSV" ]; then
    printf '%s\n' "$CSV_HEADER" 2>/dev/null > "$CSV" || die_state "cannot write its CSV"
  elif [ "$(grep '^utc,' "$CSV" 2>/dev/null | tail -n 1)" != "$CSV_HEADER" ]; then
    # A CSV whose LATEST header is another column layout: mark the boundary,
    # then this layout's header. (The latest, not the first: a file that was
    # converted once must not gain a marker on every restart.)
    printf '%s\n%s\n' "# layout change: the rows below follow this header" "$CSV_HEADER" 2>/dev/null >> "$CSV" \
      || die_state "cannot write its CSV"
  fi
  [ -w "$CSV" ] || die_state "cannot write its CSV"
  local last_drop=0 streak=0 now hr before after recovered err rc gap decision row avail free
  local fp sh mem unreadable=0 csv_said=0 since_desc
  # The start line is the first real write to the log: a log that cannot be
  # written stops the valve here.
  must_log "mem-floor-valve started (true-headroom floor ${FLOOR_MIB} MiB, interval ${INTERVAL}s, min gap ${MIN_GAP}s, max gap ${MAX_GAP}s, a drop that recovers less than ${MIN_DROPPABLE_MIB} MiB doubles the wait, reclaim=${RECLAIM})"
  while :; do
    now=$(date +%s)
    hr="$(zone_headroom_mib)"
    decision="$(valve_decide "$hr" "$(( now - last_drop ))" "$streak")"
    # The remaining columns describe the cache for reading the curve afterwards.
    set -- $(cache_mib); fp="${1:-}"; sh="${2:-}"; mem="${3:-}"
    avail="$(meminfo_mib MemAvailable)"; free="$(meminfo_mib MemFree)"
    # A blind valve says so, once, in the log a reader tails — and says when
    # it can see again.
    case "${hr#-}" in
      ''|*[!0-9]*)
        if [ "$unreadable" = 0 ]; then
          must_log "UNREADABLE: the margin cannot be read from $ZONEINFO (no Normal zone, or the file is missing); nothing is dropped until it can be read"
          unreadable=1
        fi ;;
      *)
        if [ "$unreadable" = 1 ]; then
          must_log "READABLE: the margin can be read again — true headroom ${hr} MiB"
          unreadable=0
        fi ;;
    esac
    if [ "$decision" = DROP ]; then
      # Recorded before the act, so that a helper which hangs leaves a trace.
      since_desc=""
      if [ "$last_drop" -gt 0 ]; then since_desc=" ($(( now - last_drop )) s since the last drop)"; fi
      must_log "TRY: true headroom ${hr:-?} MiB < floor ${FLOOR_MIB} MiB; dropping clean cache${since_desc}"
      before="$hr"
      err="$(valve_reclaim 2>&1 >/dev/null)"; rc=$?
      err="${err%%$'\n'*}"                      # its first line only
      err="$(printf '%s' "$err" | head -c 200)"
      after="$(zone_headroom_mib)"
      last_drop=$now
      if [ "$rc" -eq 0 ]; then
        recovered="$(valve_recovered "$before" "$after")"
        streak="$(valve_next_streak "$recovered" "$streak")"
        gap="$(valve_gap "$streak")"
        must_log "RECLAIM: true headroom ${before:-?} MiB < floor ${FLOOR_MIB} MiB -> dropped clean cache -> headroom ${after:-?} MiB (recovered ${recovered:-an unreadable amount} MiB; file pages ${fp:-?} MiB, shared memory ${sh:-?} MiB, mapped ${mem:-?} MiB; next drop allowed in ${gap} s)"
        # Any integer counts, a negative one included (a burst ate more than
        # the drop freed); only an unreadable amount is passed over.
        case "${recovered#-}" in
          ''|*[!0-9]*) ;;
          *) if [ "$recovered" -lt "$MIN_DROPPABLE_MIB" ]; then
               must_log "LOW: that drop recovered less than ${MIN_DROPPABLE_MIB} MiB — the margin is held by what the processes themselves have, not by clean file cache; waiting ${gap} s before the next try"
             fi ;;
        esac
        row="$hr,$fp,$sh,$mem,$avail,$free,DROPPED->${after:-?}"
      else
        # A failed drop starts the wait and lengthens it: a helper that is
        # missing or not granted is not retried every reading.
        streak=$(( streak + 1 )); gap="$(valve_gap "$streak")"
        must_log "RECLAIM FAILED at true headroom ${before:-?} MiB (exit ${rc}${err:+; $err}): is ${RECLAIM} installed, executable and granted in sudoers? Run it by hand to see its error. Next try in ${gap} s"
        row="$hr,$fp,$sh,$mem,$avail,$free,DROP-FAILED"
      fi
    else
      row="$hr,$fp,$sh,$mem,$avail,$free,$decision"
    fi
    # The CSV is the curve, not the record of acts: if it cannot be written
    # the valve keeps working, and says so once.
    if ! printf '%s,%s\n' "$(date -u +%FT%TZ)" "$row" 2>/dev/null >> "$CSV"; then
      if [ "$csv_said" = 0 ]; then
        must_log "CSV: a reading could not be appended to $CSV; the log still records every drop"
        csv_said=1
      fi
    fi
    sleep "$INTERVAL"
  done
}

if [ "${VALVE_LIB_ONLY:-0}" = "1" ]; then return 0 2>/dev/null || exit 0; fi

# A setting that is not a whole number would make a comparison error out, and
# an errored comparison reads as false — which could let a drop through. The
# same holds for a number too large for the shell's arithmetic, so settings
# are capped at nine digits, far below it.
for setting in "$FLOOR_MIB" "$INTERVAL" "$MIN_GAP" "$MAX_GAP" "$MIN_DROPPABLE_MIB"; do
  case "$setting" in
    '') printf 'mem-floor-valve: FLOOR_MIB, INTERVAL, MIN_GAP, MAX_GAP and MIN_DROPPABLE_MIB must be whole numbers\n' >&2; exit 2 ;;
    *[!0-9]*) printf 'mem-floor-valve: FLOOR_MIB, INTERVAL, MIN_GAP, MAX_GAP and MIN_DROPPABLE_MIB must be whole numbers (got "%s")\n' "$setting" >&2; exit 2 ;;
    ??????????*) printf 'mem-floor-valve: FLOOR_MIB, INTERVAL, MIN_GAP, MAX_GAP and MIN_DROPPABLE_MIB must be at most nine digits (got "%s")\n' "$setting" >&2; exit 2 ;;
  esac
done
if [ "$INTERVAL" -eq 0 ] || [ "$MIN_GAP" -eq 0 ] || [ "$MAX_GAP" -eq 0 ]; then
  printf 'mem-floor-valve: INTERVAL, MIN_GAP and MAX_GAP must be above 0\n' >&2; exit 2
fi
if [ "$MAX_GAP" -lt "$MIN_GAP" ]; then
  printf 'mem-floor-valve: MAX_GAP must be at least MIN_GAP\n' >&2; exit 2
fi
# Leading zeros off, so that nothing downstream can read a setting as octal.
FLOOR_MIB=$((10#$FLOOR_MIB));      INTERVAL=$((10#$INTERVAL))
MIN_GAP=$((10#$MIN_GAP));          MAX_GAP=$((10#$MAX_GAP))
MIN_DROPPABLE_MIB=$((10#$MIN_DROPPABLE_MIB))

if [ "$#" -eq 1 ] && [ "${1:-}" = "--status" ]; then
  # One reading, and what a valve whose wait is open would decide. This does
  # not know when the running valve last dropped; its log does.
  hr="$(zone_headroom_mib)"
  set -- $(cache_mib)
  decision="$(valve_decide "$hr" "$MIN_GAP" 0)"
  printf 'true headroom %s MiB (floor %s); clean file pages %s MiB, mapped in total %s MiB, shared memory %s MiB -> %s\n' \
    "${hr:-UNREADABLE}" "$FLOOR_MIB" "${1:-UNREADABLE}" "${3:-UNREADABLE}" "${2:-UNREADABLE}" "$decision"
  exit 0
fi
if [ "$#" -ne 0 ]; then
  printf 'usage: mem-floor-valve.sh [--status]\n' >&2; exit 2
fi

valve_loop
