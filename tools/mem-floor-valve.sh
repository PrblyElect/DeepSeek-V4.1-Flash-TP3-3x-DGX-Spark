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
# That back-off is lifted under NO_THROTTLE_MIB: there the wait stays MIN_GAP
# whatever the streak, because at that margin a drop can still give back far
# more than the ones before it — until a try FAILS, which brings the back-off
# back. 0 turns the lift off.
#
# WHAT IT DOES NOT DO
# It never stops, starts, signals or inspects the engine's container; its only
# privileged act is one call to a helper that drops clean cache. It never
# drops on a margin it could not read. It cannot create memory: when the
# margin is low because of what the processes themselves hold, a drop
# recovers little, and the valve says so in its log and waits longer — unless
# the margin is under NO_THROTTLE_MIB, where it keeps trying at the minimum
# wait. And it does nothing it cannot record: it stops rather than run without
# its log.
#
# Settings (environment variables; the defaults are the shipped values):
#   FLOOR_MIB=1000                          drop below this much true headroom
#   INTERVAL=10                             seconds between readings
#   MIN_GAP=30                              the wait after a drop that recovered
#                                           at least MIN_DROPPABLE_MIB
#   MAX_GAP=600                             the wait doubles after each drop that
#                                           recovered less, up to this
#   MIN_DROPPABLE_MIB=128                   what a drop must recover to count
#   NO_THROTTLE_MIB=300                     under this much true headroom the back-off
#                                           does not apply: the wait is MIN_GAP whatever
#                                           the streak, until a try fails. 0 turns it off.
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
NO_THROTTLE_MIB="${NO_THROTTLE_MIB:-300}"
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
#   (consecutive drops that recovered less than MIN_DROPPABLE_MIB)
#   $4 optional, 1 when the last try FAILED.
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
  # false, which is a DROP. The gap is valve_wait's for this margin: the
  # streak's, or the minimum where the back-off is lifted.
  gap="$(valve_wait "$hr" "$streak" "${4:-0}")"
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
# The back-off (valve_gap: the wait doubles per poor drop) keeps a drop that
# gives back little from being repeated often. Its cost: after a run of poor
# drops the valve waits out its longest gap at the very margins where a drop
# still gives back the most. Under NO_THROTTLE_MIB that back-off is lifted.
# valve_lifted $1 headroom_mib -> status 0 where the back-off is lifted at
# this margin: NO_THROTTLE_MIB above 0, and the margin under it. Anything
# unreadable, the margin included, is never lifted.
valve_lifted() {
  case "${NO_THROTTLE_MIB:-}" in ''|*[!0-9]*|??????????*) return 1 ;; esac
  case "${1#-}" in ''|*[!0-9]*) return 1 ;; esac
  [ "$((10#$NO_THROTTLE_MIB))" -gt 0 ] && [ "$1" -lt "$((10#$NO_THROTTLE_MIB))" ]
}
# valve_wait $1 headroom_mib  $2 the streak  $3 1 when the last try FAILED
#   -> the wait before the next drop, in seconds; EMPTY if it cannot be told.
#   One function for the decision and for the log, so that the log states the
#   wait the decision enforces — and, where that depends on which side of the
#   line the NEXT margin falls, says so (the loop writes both). Where the
#   back-off is lifted the wait is the minimum whatever the streak; a drop is
#   thus still at most one per
#   MIN_GAP, and the streak keeps counting — it applies again as soon as the
#   margin rises above NO_THROTTLE_MIB. After a failed try the back-off is
#   never lifted: a helper that is missing or not granted must not be retried
#   every MIN_GAP.
valve_wait() {
  local gap; gap="$(valve_gap "$2")"
  case "$gap" in ''|*[!0-9]*) return ;; esac
  if [ "${3:-0}" != 1 ] && valve_lifted "$1"; then
    gap="$(valve_gap 0)"                # never the raw setting: the same checks as above
    case "$gap" in ''|*[!0-9]*) return ;; esac
  fi
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
  local last_drop=0 streak=0 try_failed=0 now hr before after recovered err rc gap decision row avail free
  local fp sh mem unreadable=0 csv_said=0 since_desc lift_desc gap_desc low_desc lift_note
  # The back-off line as set: a positive value is named; 0 says it is off.
  lift_note="no back-off under ${NO_THROTTLE_MIB} MiB"
  if ! valve_lifted 0; then lift_note="the back-off lifted nowhere"; fi     # 0, or not a number: the lift is off
  # The start line is the first real write to the log: a log that cannot be
  # written stops the valve here.
  must_log "mem-floor-valve started (true-headroom floor ${FLOOR_MIB} MiB, interval ${INTERVAL}s, min gap ${MIN_GAP}s, max gap ${MAX_GAP}s, a drop that recovers less than ${MIN_DROPPABLE_MIB} MiB doubles the wait, ${lift_note}, reclaim=${RECLAIM})"
  while :; do
    now=$(date +%s)
    hr="$(zone_headroom_mib)"
    decision="$(valve_decide "$hr" "$(( now - last_drop ))" "$streak" "$try_failed")"
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
      # Said where it is in force, so the back-off's absence is on record too.
      lift_desc=""
      if [ "$try_failed" != 1 ] && valve_lifted "$hr"; then lift_desc=" (no back-off under ${NO_THROTTLE_MIB} MiB)"; fi
      must_log "TRY: true headroom ${hr:-?} MiB < floor ${FLOOR_MIB} MiB; dropping clean cache${since_desc}${lift_desc}"
      before="$hr"
      err="$(valve_reclaim 2>&1 >/dev/null)"; rc=$?
      err="${err%%$'\n'*}"                      # its first line only
      err="$(printf '%s' "$err" | head -c 200)"
      after="$(zone_headroom_mib)"
      last_drop=$now
      if [ "$rc" -eq 0 ]; then
        try_failed=0
        recovered="$(valve_recovered "$before" "$after")"
        streak="$(valve_next_streak "$recovered" "$streak")"
        gap="$(valve_wait "$after" "$streak" 0)"
        # Which wait the next decision enforces depends on the NEXT reading's
        # margin, not on this one's. So a result line never states one wait as
        # if it held everywhere: under the line, it says the wait holds only
        # while the margin stays there; above it, where the wait is longer, it
        # names the minimum that applies once the margin falls under the line.
        gap_desc=""; low_desc=""
        if [ -n "$gap" ]; then
          if valve_lifted "$after"; then
            gap_desc=" while the margin stays under ${NO_THROTTLE_MIB} MiB"
          elif valve_lifted 0 && [ "$gap" != "$(valve_gap 0)" ]; then      # the lift is on, and the waits differ
            gap_desc=", or $(valve_gap 0) s once the margin is under ${NO_THROTTLE_MIB} MiB"
            low_desc=" — the minimum, $(valve_gap 0) s, once the margin is under ${NO_THROTTLE_MIB} MiB"
          fi
        fi
        must_log "RECLAIM: true headroom ${before:-?} MiB < floor ${FLOOR_MIB} MiB -> dropped clean cache -> headroom ${after:-?} MiB (recovered ${recovered:-an unreadable amount} MiB; file pages ${fp:-?} MiB, shared memory ${sh:-?} MiB, mapped ${mem:-?} MiB; next drop allowed in ${gap} s${gap_desc})"
        # Any integer counts, a negative one included (a burst ate more than
        # the drop freed); only an unreadable amount is passed over.
        case "${recovered#-}" in
          ''|*[!0-9]*) ;;
          *) if [ "$recovered" -lt "$MIN_DROPPABLE_MIB" ]; then
               # Under the line a poor drop every MIN_GAP is the design, not a
               # diagnosis of what holds the margin: the classic sentence is
               # for the other case, where it is what the back-off rests on.
               # Under the line only what is true by construction is said: the
               # drop gave back little (a burst may even have eaten more than it
               # freed, which reads as a negative recovery).
               if [ -n "$gap" ] && valve_lifted "$after"; then
                 must_log "LOW: that drop recovered less than ${MIN_DROPPABLE_MIB} MiB — that drop gave back little; waiting ${gap} s before the next try — while the margin stays under ${NO_THROTTLE_MIB} MiB"
               else
                 must_log "LOW: that drop recovered less than ${MIN_DROPPABLE_MIB} MiB — the margin is held by what the processes themselves have, not by clean file cache; waiting ${gap} s before the next try${low_desc}"
               fi
             fi ;;
        esac
        row="$hr,$fp,$sh,$mem,$avail,$free,DROPPED->${after:-?}"
      else
        # A failed drop starts the wait and lengthens it: a helper that is
        # missing or not granted is not retried every reading. It also turns
        # the lift off until a drop succeeds.
        streak=$(( streak + 1 )); try_failed=1; gap="$(valve_wait "$before" "$streak" 1)"
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
for setting in "$FLOOR_MIB" "$INTERVAL" "$MIN_GAP" "$MAX_GAP" "$MIN_DROPPABLE_MIB" "$NO_THROTTLE_MIB"; do
  case "$setting" in
    '') printf 'mem-floor-valve: FLOOR_MIB, INTERVAL, MIN_GAP, MAX_GAP, MIN_DROPPABLE_MIB and NO_THROTTLE_MIB must be whole numbers\n' >&2; exit 2 ;;
    *[!0-9]*) printf 'mem-floor-valve: FLOOR_MIB, INTERVAL, MIN_GAP, MAX_GAP, MIN_DROPPABLE_MIB and NO_THROTTLE_MIB must be whole numbers (got "%s")\n' "$setting" >&2; exit 2 ;;
    ??????????*) printf 'mem-floor-valve: FLOOR_MIB, INTERVAL, MIN_GAP, MAX_GAP, MIN_DROPPABLE_MIB and NO_THROTTLE_MIB must be at most nine digits (got "%s")\n' "$setting" >&2; exit 2 ;;
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
NO_THROTTLE_MIB=$((10#$NO_THROTTLE_MIB))

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
