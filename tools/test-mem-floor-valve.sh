#!/usr/bin/env bash
# test-mem-floor-valve.sh — desk test for tools/mem-floor-valve.sh.
#
# Needs bash, awk, sed, grep and perl (a portable timeout) — and ps, for one
# case. Needs no box, no root and no network, and is safe on a serving node. The
# decision is a set of pure functions; the readers are run against fixture
# files; and the loop runs with its readings and its privileged call replaced.
# Nothing here can reach the kernel: the helper the valve is given is a
# stand-in that only appends a line to a file (or, in one case, puts the valve
# itself out of its misery), `sudo` is a shell function that runs that
# stand-in and nothing else — exported into every run as a program too — and
# the valve's readings are always fixture files. So it stays safe on any box,
# whatever that box's sudoers allows, and whether or not the box has a /proc.
#
# The fixtures are round numbers chosen for the arithmetic, never readings
# captured from a node.
#
#   bash tools/test-mem-floor-valve.sh
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VALVE="$HERE/mem-floor-valve.sh"
TMP="$(mktemp -d)" || { echo "test-mem-floor-valve: mktemp failed — no scratch directory" >&2; exit 1; }
chmod 700 "$TMP"
trap 'rm -rf "$TMP"' EXIT
export STATE_DIR="$TMP/state"          # the imported functions take their paths from this
pass=0; fail=0
ok()    { printf '  PASS  %s\n' "$1"; pass=$((pass + 1)); }
bad()   { printf '  FAIL  %s\n' "$1"; fail=$((fail + 1)); }
check() { if [ "$2" -eq 0 ]; then ok "$1"; else bad "$1 — $3"; fi; }
part()  { printf '\n%s\n' "$1"; }
# `perl -e 'alarm N; exec @ARGV'` is a portable timeout: whatever should have
# stopped, the alarm ends it and the exit code says so.
within() { local secs="$1"; shift; perl -e 'alarm shift; exec @ARGV' "$secs" "$@"; }

# ---- fixtures, which the valve as a program can be pointed at too -------------
# All round: 398336 − 65536 pages = 332800 × 4 KiB = 1300 MiB of true headroom.
ZONE="$TMP/zoneinfo"
printf '%s\n' 'Node 0, zone   DMA' '  pages free     100000' '        min      0' \
  'Node 0, zone   Normal' '  pages free     398336' '        boost    0' '        min      65536' \
  '        low      98304' '      nr_free_pages 999999' 'Node 0, zone  Movable' '  pages free     0' '        min      0' > "$ZONE"
# clean file pages 600 + 340 = 940 MiB; shared memory 820 MiB; mapped 1690 MiB
MEM="$TMP/meminfo"
printf '%s\n' 'MemTotal:       131072000 kB' 'MemFree:          8388608 kB' 'MemAvailable:    20971520 kB' \
  'Cached:         26214400 kB' 'Active(file):     614400 kB' 'Inactive(file):   348160 kB' \
  'Shmem:            839680 kB' 'Mapped:          1730560 kB' > "$MEM"
export VALVE_ZONEINFO="$ZONE" VALVE_MEMINFO="$MEM"
FAKE_RECLAIM="$TMP/fake-reclaim"       # written per scenario: records the call, says and exits as told
SUDO_CALLS="$TMP/sudo.calls"
RECLAIM_CALLS="$TMP/reclaim.calls"
export SUDO_CALLS RECLAIM_CALLS

# ---- the valve as a library ---------------------------------------------------
# The functions are read out of the valve itself, never copied here: a test
# with its own copy of the decision only tests itself.
part "the valve as a library"
FUNCS="valve_decide valve_gap valve_recovered valve_next_streak zone_headroom_mib cache_mib meminfo_mib valve_loop log must_log valve_reclaim"
IMPORT="$(within 5 env VALVE_LIB_ONLY=1 bash -c 'source "$1" && declare -f '"$FUNCS" _ "$VALVE" 2>/dev/null)"
CSV_HEADER="$(within 5 env VALVE_LIB_ONLY=1 bash -c 'source "$1" && printf "%s" "$CSV_HEADER"' _ "$VALVE" 2>/dev/null)"
# The loop names its input files in its log lines: take the valve's own values.
eval "$(within 5 env VALVE_LIB_ONLY=1 bash -c 'source "$1" && printf "ZONEINFO=%q\nMEMINFO=%q\n" "$ZONEINFO" "$MEMINFO"' _ "$VALVE" 2>/dev/null)"
missing=""
if [ -n "$IMPORT" ]; then eval "$IMPORT"; fi
for f in $FUNCS; do declare -F "$f" >/dev/null 2>&1 || missing="$missing $f"; done
[ -z "$missing" ] && [ -n "$CSV_HEADER" ]
check "sourced with VALVE_LIB_ONLY=1, it defines every function and its header, and does not run" $? "missing:$missing"
FLOOR_MIB=1000; INTERVAL=0; MIN_GAP=30; MAX_GAP=600; MIN_DROPPABLE_MIB=128

decides() {   # $1 the case  $2 expected  $3 headroom  $4 seconds since the last drop  $5 the streak
  local got; got="$(valve_decide "$3" "$4" "$5" 2>/dev/null)"
  [ "$got" = "$2" ]; check "$1 -> $2" $? "got '$got'"
}
is() {       # $1 the case  $2 got  $3 want
  [ "$2" = "$3" ]; check "$1 (want $3)" $? "got '$2'"
}
empty() {    # $1 the case  $2 got
  [ -z "$2" ]; check "$1 (want EMPTY)" $? "got '$2'"
}

part "the decision: under the floor, and not before"
decides "a margin of 100 MiB, long since the last drop"      DROP               100  999  0
decides "a margin of a few dozen MiB"                        DROP               40   31   0
decides "free below the reserve line (-5)"                   DROP               -5   999  0
decides "1 MiB under the floor"                              DROP               999  999  0
decides "plenty of headroom (does not trip on a healthy node)" "WAIT:headroom-ok" 1800 999 0
decides "exactly at the floor"                               "WAIT:headroom-ok" 1000 999 0
decides "above the floor even deep in a back-off"            "WAIT:headroom-ok" 1001 999 9
decides "unreadable headroom"                                "WAIT:unreadable"  ""   999  0
decides "non-numeric headroom"                               "WAIT:unreadable"  abc  999  0
decides "a margin of zero (written -0 by some awks) is under the floor" DROP          -0   999  0
decides "under the floor, unreadable clock"                  "WAIT:throttled"   100  ""   0
decides "under the floor, non-numeric clock"                 "WAIT:throttled"   100  soon 0
decides "under the floor, unreadable streak counts as none"  DROP               100  999  abc

part "the decision: the wait after a drop"
decides "29 s after a good drop"                                   "WAIT:throttled" 100 29  0
decides "30 s after a good drop"                                   DROP             100 30  0
decides "29 s after a drop that recovered little"                  "WAIT:throttled" 100 29  1
decides "30 s after a drop that recovered little: still waiting, the wait doubled" "WAIT:throttled" 100 30 1
decides "60 s after a drop that recovered little"                  DROP             100 60  1
decides "two poor drops: 119 s"                                    "WAIT:throttled" 100 119 2
decides "two poor drops: 120 s"                                    DROP             100 120 2
decides "three poor drops: 240 s"                                  DROP             100 240 3
decides "four poor drops: 480 s"                                   DROP             100 480 4
decides "five poor drops: the wait is capped, 599 s"               "WAIT:throttled" 100 599 5
decides "five poor drops: 600 s"                                   DROP             100 600 5
decides "nine poor drops: still 600 s, a drop is never abandoned"   DROP             100 600 9

part "the decision: the wait and the streak, as functions"
is  "the wait after no poor drop"                        "$(valve_gap 0)"    30
is  "the wait after one poor drop"                       "$(valve_gap 1)"    60
is  "the wait after two poor drops"                      "$(valve_gap 2)"    120
is  "the wait after three poor drops"                    "$(valve_gap 3)"    240
is  "the wait after four poor drops"                     "$(valve_gap 4)"    480
is  "the wait after five poor drops (capped)"            "$(valve_gap 5)"    600
is  "the wait after twenty poor drops"                   "$(valve_gap 20)"   600
is  "an unreadable streak waits the minimum"             "$(valve_gap abc)"  30
save_min="$MIN_GAP"; save_max="$MAX_GAP"
MIN_GAP=0090 MAX_GAP=0600
is  "a zero-padded MIN_GAP is decimal: no poor drop -> 90"   "$(valve_gap 0)"    90
is  "  one poor drop -> 180"                                 "$(valve_gap 1)"    180
is  "  and the zero-padded cap holds at 600"                 "$(valve_gap 9)"    600
MIN_GAP=0010 MAX_GAP=0050
is  "  (0010 is ten: one poor drop -> 20; octal would say 16)" "$(valve_gap 1)"    20
is  "  (0050 is fifty: the cap holds there, not at 40)"         "$(valve_gap 3)"    50
MIN_GAP=abc MAX_GAP=600
got="$(valve_gap 0)"; empty "a minimum gap that is not a whole number -> EMPTY (the caller waits)" "$got"
decides "  and the decision waits on it, whatever the clock" "WAIT:throttled" 100 999  0
MIN_GAP=30 MAX_GAP=abc
got="$(valve_gap 0)"; empty "a maximum gap that is not a whole number -> EMPTY too (checked on its own)" "$got"
decides "  and the decision waits on that as well" "WAIT:throttled" 100 999  0
MIN_GAP=30 MAX_GAP=""
got="$(valve_gap 0)"; empty "an empty maximum gap -> EMPTY (an empty one must not hide behind a readable one)" "$got"
# The minimum's own check, by the only case that tells it from the arithmetic:
# an EMPTY minimum is 0 to `$((10#))` on bash 3.2 — a gap of 0 is no wait.
MIN_GAP="" MAX_GAP=600
got="$(valve_gap 0)"; empty "an empty minimum gap -> EMPTY (the arithmetic alone would make it 0: no wait)" "$got"
decides "  and the decision waits on it" "WAIT:throttled" 100 0 0
decides "  whatever the clock says" "WAIT:throttled" 100 999 0
# Platform note for the mutation check: with the cap's own check REMOVED, this
# still comes back EMPTY on bash 5 — an arithmetic error there ends the
# command substitution — and only bash 3.2 carries on and returns a number.
# So that mutation is caught here on bash 3.2 and is invisible on bash 5,
# where it fails closed through the decision's empty-gap guard instead.
MIN_GAP="$save_min"; MAX_GAP="$save_max"
is  "a good drop resets the streak (400 MiB recovered)"  "$(valve_next_streak 400 3)" 0
is  "a drop at the threshold counts as good (128 MiB)"   "$(valve_next_streak 128 3)" 0
is  "a poor drop adds one (127 MiB)"                     "$(valve_next_streak 127 3)" 4
is  "a poor drop from none"                              "$(valve_next_streak 50 0)"  1
is  "a burst that ate more than the drop freed counts as poor (-5 MiB)" "$(valve_next_streak -5 2)" 3
is  "an unreadable amount counts as poor"                "$(valve_next_streak "" 1)"  2
is  "recovered: 900 -> 1500"                             "$(valve_recovered 900 1500)" 600
is  "recovered: 900 -> 895 is negative"                  "$(valve_recovered 900 895)"   -5
got="$(valve_recovered "" 10 2>/dev/null)";  empty "recovered: an unreadable before" "$got"
got="$(valve_recovered abc 10 2>/dev/null)"; empty "recovered: a non-numeric before (the arithmetic alone would read it as 0)" "$got"
got="$(valve_recovered 10 xyz 2>/dev/null)"; empty "recovered: a non-numeric after" "$got"
# Under `set -u` the arithmetic alone fails closed on a NAME it cannot see; what
# it would not fail on is the name of a number it can. This is why each side
# is checked before any arithmetic.
got="$(valve_recovered FLOOR_MIB 1010 2>/dev/null)"; empty "recovered: a 'before' that is the name of a number in scope (the arithmetic alone would return 10)" "$got"
got="$(valve_recovered 10 "" 2>/dev/null)";  empty "recovered: an unreadable after"  "$got"

part "the readers, against fixture files"
got="$(zone_headroom_mib "$ZONE")";  [ "$got" = 1300 ]; check "headroom: reads the Normal zone, not the zone before or after, and not nr_free_pages (want 1300)" $? "got '$got'"
sed 's/pages free     398336/pages free      60000/' "$ZONE" > "$ZONE.neg"
got="$(zone_headroom_mib "$ZONE.neg")"; [ "$got" = -21 ]; check "headroom: free below the reserve line is a NEGATIVE reading" $? "got '$got'"
printf '%s\n' 'Node 0, zone   DMA' '  pages free     100000' '        min      0' > "$ZONE.nonormal"
got="$(zone_headroom_mib "$ZONE.nonormal")"; empty "headroom: no Normal zone (never a silent 0)" "$got"
: > "$ZONE.empty"
got="$(zone_headroom_mib "$ZONE.empty")";    empty "headroom: an empty file (never a silent 0)" "$got"
got="$(cache_mib "$MEM")"; [ "$got" = "940 820 1690" ]; check "cache: clean file pages 940, shared memory 820, mapped 1690, as three separate quantities" $? "got '$got'"
printf '%s\n' 'Active(file):      102400 kB' 'Inactive(file):     102400 kB' 'Shmem:            2560000 kB' 'Mapped:           4096000 kB' > "$MEM.moreshmem"
got="$(cache_mib "$MEM.moreshmem")"; [ "$got" = "200 2500 4000" ]; check "cache: more shared memory than file pages is an ordinary state, and file pages are NOT reduced by it (shared memory is not in them)" $? "got '$got'"
printf '%s\n' 'MemTotal:       131072000 kB' 'Active(file):     614400 kB' 'Inactive(file):   348160 kB' 'Shmem:            839680 kB' > "$MEM.short"
got="$(cache_mib "$MEM.short")"; empty "cache: a missing key (Mapped here) (never a silent 0)" "$got"
got="$(meminfo_mib MemAvailable "$MEM")"; [ "$got" = 20480 ]; check "one meminfo key (MemAvailable 20480 MiB)" $? "got '$got'"
got="$(meminfo_mib NoSuchKey "$MEM")";    empty "one meminfo key: missing" "$got"

# ---- the loop, with its readings and its privileged call replaced -------------
LOGF="$TMP/state/mem-floor.log"; CSVF="$TMP/state/mem-floor.csv"
# A reading sequence that survives command substitution. The loop takes every
# reading inside `$(...)`, a subshell, so a shell-variable counter cannot
# advance; the counter lives in a file.
next() {   # $1 counter file  $2 sequence file (one reading per line; the last one repeats)
  local n total
  n=$(( $(cat "$1" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$1"
  total=$(wc -l < "$2" | tr -d ' ')
  [ "$n" -gt "$total" ] && n=$total
  sed -n "${n}p" "$2"
}
# The privileged call, exactly as the valve must make it: `sudo -n <helper>`.
# This records that sudo was asked, then runs the stand-in and nothing else. A
# valve that called the helper directly would leave no record. Exported, so a
# run as a program goes through it too.
sudo() { printf 'sudo\n' >> "$SUDO_CALLS"; shift; "$@"; }
export -f sudo
# A non-root valve must go through sudo; root runs the helper directly.
if [ "$(id -u)" -eq 0 ]; then want_sudo=0; else want_sudo=1; fi
run_loop() {
  (
    # This shell's own PID, for the watchdog below. Not `$!` from outside: a
    # subshell is forked before any such assignment and never sees it. (bash
    # 3.2 has no $BASHPID; a shell exec'd inside `$(...)` has this shell for
    # its parent.) Without the PID the loop is not run at all.
    LOOP_PID="$(exec sh -c 'echo "$PPID"')"
    [ -n "$LOOP_PID" ] || { echo "HARNESS: the loop's shell does not know its own PID; the loop is not run" >&2; exit 98; }
    echo "$LOOP_PID" > "$TMP/loop-pid"
    LOG="$LOGF"; CSV="$CSVF"; RECLAIM="$FAKE_RECLAIM"
    INTERVAL=0; FLOOR_MIB=1000; MIN_GAP=30; MAX_GAP=600; MIN_DROPPABLE_MIB=128
    eval "${VALVE_OPTS:-}"
    zone_headroom_mib() {
      # Watchdog. Each drop costs one extra reading, so a $READINGS-reading
      # scenario can never legitimately need more than this many. A loop that
      # does not stop (say, with the valve's sleep removed) is ENDED here.
      # This stub runs inside `$(...)`: an `exit` here would end only that
      # substitution and the loop would go on — so it ends the loop's shell.
      if [ "$(cat "$TMP/n-hr" 2>/dev/null || echo 0)" -gt $(( READINGS * 2 + 4 )) ]; then
        printf 'HARNESS: the loop did not stop — more than %s readings in a %s-reading scenario; ended by the harness\n' \
          $(( READINGS * 2 + 4 )) "$READINGS" >&2
        : > "$TMP/watchdog"
        kill -9 "$LOOP_PID"
      fi
      if [ -e "$TMP/csv-locked" ] && [ ! -e "$TMP/csv-locked.done" ]; then        # (one scenario) lock the CSV after start
        : > "$TMP/csv-locked.done"; chmod 444 "$CSVF"
      fi
      local v; v="$(next "$TMP/n-hr" "$TMP/seq-hr")"; [ "$v" = x ] || echo "$v"; }   # x = an unreadable reading
    cache_mib() { echo "900 800 1600"; }
    meminfo_mib() { echo 0; }
    # The loop ends on its READINGS'th sleep. SLEEP_STUB=off removes that end,
    # for the watchdog's own check.
    if [ "${SLEEP_STUB:-}" = off ]; then sleep() { :; }
    else s_n=0; sleep() { s_n=$((s_n + 1)); [ "$s_n" -lt "$READINGS" ] || exit 0; }; fi
    valve_loop
  )
}
scenario() {   # KEY=VALUE ... (quote values that contain spaces); starts from a clean slate
  rm -rf "$SUDO_CALLS" "$RECLAIM_CALLS" "$TMP/n-hr" "$TMP/watchdog"
  SEQ_HR=""; HELPER_RC=0; HELPER_SAYS=""; READINGS=3; VALVE_OPTS=""; KEEP_STATE=0; NO_CSV_ROWS=0; SLEEP_STUB=""
  local kv; for kv in "$@"; do eval "$kv"; done
  # After the arguments, not before: KEEP_STATE is one of them.
  chmod -R u+w "$TMP/state" 2>/dev/null
  [ "$KEEP_STATE" = 1 ] || rm -rf "$TMP/state"
  # shellcheck disable=SC2086  # word-splitting is the point: one reading per line
  printf '%s\n' $SEQ_HR > "$TMP/seq-hr"
  printf '#!/bin/sh\nprintf "ran\\n" >> "%s"\nprintf "%%s\\n" "%s" >&2\nexit %s\n' "$RECLAIM_CALLS" "$HELPER_SAYS" "$HELPER_RC" > "$FAKE_RECLAIM"
  chmod +x "$FAKE_RECLAIM"
  rows_before="$(data_rows)"
  if [ -n "${LOOP_ERR:-}" ]; then run_loop 2> "$LOOP_ERR"; else run_loop; fi
  # The loop must have taken its readings, or every check below passes for
  # nothing (a loop that never ran makes "no drop attempted" true as well).
  # The proof does not depend on the CSV: the reading stub is counted, and
  # each drop costs exactly one extra reading (the margin afterwards).
  stub_calls="$(cat "$TMP/n-hr" 2>/dev/null || echo 0)"; drops="$(count x "$RECLAIM_CALLS")"
  [ "$(( stub_calls - drops ))" -eq "$READINGS" ]
  check "  (harness) the loop took $READINGS readings" $? "stub calls $stub_calls, drops $drops"
  if [ -e "$TMP/watchdog" ]; then
    check "  (harness) the loop stopped by itself" 1 "ended by the watchdog after $stub_calls readings in a $READINGS-reading scenario"
  fi
  if [ "$NO_CSV_ROWS" != 1 ]; then
    rows=$(( $(data_rows) - rows_before ))
    [ "$rows" -eq "$READINGS" ]
    check "  (harness) and wrote them to the CSV" $? "the CSV gained $rows data rows"
  fi
}
count()    { if [ -f "$2" ]; then wc -l < "$2" | tr -d ' '; else echo 0; fi; }
data_rows() { if [ -f "$CSVF" ]; then grep -vc '^utc,\|^# ' "$CSVF"; else echo 0; fi; }
log_has()   { if [ -f "$LOGF" ]; then grep -c -- "$1" "$LOGF" || true; else echo "no log"; fi; }
log_lines() { wc -l < "$LOGF" | tr -d ' '; }
log_order() {   # true if the first line matching $1 comes before the first line matching $2
  local a b; a=$(grep -n -- "$1" "$LOGF" | head -n 1 | cut -d: -f1); b=$(grep -n -- "$2" "$LOGF" | head -n 1 | cut -d: -f1)
  [ -n "$a" ] && [ -n "$b" ] && [ "$a" -lt "$b" ]
}
decisions() { grep -v '^utc,\|^# ' "$CSVF" | cut -d, -f8 | tr '\n' ' '; }
expect() { [ "$2" = "$3" ]; check "$1" $? "got '$2', want '$3'"; }
expect_has() { grep -q -- "$2" "$LOGF"; check "$1" $? "the log lacks: $2"; }

part "the loop (readings and the privileged call replaced)"
# The watchdog itself, on a loop that cannot end: the valve's sleep is
# replaced by a no-op, so only the watchdog can end it. The checks inside
# that run are recorded, not counted — the harness check that fails there is
# the one that has to. This run is a job with a bound of its own, so that a
# broken watchdog fails here instead of hanging the desk test. (The first
# watchdog said `exit 99` from inside `$(...)`, which ended only the
# substitution; the second did not know which shell to end. Both ran on.)
rm -f "$TMP/selftest-checks" "$TMP/selftest.done" "$TMP/loop-pid" "$TMP/selftest.err"
( check() { printf '%s|%s|%s\n' "$2" "$1" "$3" >> "$TMP/selftest-checks"; }
  scenario 'SEQ_HR="1500"' READINGS=2 SLEEP_STUB=off </dev/null; : > "$TMP/selftest.done" ) 2> "$TMP/selftest.err" &
pid=$!; n=0
while [ ! -e "$TMP/selftest.done" ] && [ "$n" -lt 30 ]; do sleep 1; n=$((n + 1)); done
if [ -e "$TMP/selftest.done" ]; then hung=0
else hung=1; kill -9 "$pid" "$(cat "$TMP/loop-pid" 2>/dev/null)" 2>/dev/null; fi   # (the job is done when it says so: `kill -0` is true of a zombie)
wait "$pid" 2>/dev/null
got="$(tr '\n' ';' < "$TMP/selftest-checks" 2>/dev/null | cut -c1-260)"
[ "$hung" = 0 ] && grep -q '^1|  (harness) the loop stopped by itself|ended by the watchdog after [0-9]* readings in a 2-reading scenario$' "$TMP/selftest-checks"
check "watchdog: a loop that cannot end is ended by it, and the harness says so" $? "$( [ "$hung" = 1 ] && printf 'NOT ended in %s s; ' "$n" )recorded: $got"
! grep -q '^0|  (harness) the loop took 2 readings|' "$TMP/selftest-checks"
check "  … and it is not passed off as a loop that took its readings" $? "recorded: $got"
[ "$(grep -c '^HARNESS: the loop did not stop' "$TMP/selftest.err" 2>/dev/null)" = 1 ]
check "  … and it says so once, on standard error" $? "stderr: $(head -c 200 "$TMP/selftest.err" 2>/dev/null | tr '\n' '|')"
scenario 'SEQ_HR="900 1500 1500 1500"' HELPER_RC=0 READINGS=4
expect "low, and the drop recovers a lot: one drop"         "$(count x "$RECLAIM_CALLS")" "1"
expect "  through sudo (a non-root valve must; root runs the helper itself)" "$(count x "$SUDO_CALLS")" "$want_sudo"
expect "  the margin is back above the floor, so it waits"  "$(decisions)" "DROPPED->1500 WAIT:headroom-ok WAIT:headroom-ok WAIT:headroom-ok "
expect_has "  the log records before, after and the recovery" "RECLAIM: true headroom 900 MiB < floor 1000 MiB -> dropped clean cache -> headroom 1500 MiB (recovered 600 MiB; file pages 900 MiB, shared memory 800 MiB, mapped 1600 MiB; next drop allowed in 30 s)"
expect "  and a good drop is not called LOW"                 "$(log_has 'LOW:')" "0"
expect "  the CSV carries this layout's header"              "$(head -n 1 "$CSVF")" "$CSV_HEADER"
log_order "TRY: true headroom 900 MiB" "RECLAIM: true headroom 900"
check "  and the attempt is recorded before the act" $? "no TRY line ahead of the RECLAIM line"
expect "  in three lines: start, attempt, result"            "$(log_lines)" "3"

scenario 'SEQ_HR="800 850 800 850 800 850"' HELPER_RC=0 READINGS=4
expect "a drop that recovers little: one drop, not one per reading" "$(count x "$RECLAIM_CALLS")" "1"
expect "  then it waits"                                      "$(decisions)" "DROPPED->850 WAIT:throttled WAIT:throttled WAIT:throttled "
expect_has "  the log says what the drop recovered"           "RECLAIM: true headroom 800 MiB < floor 1000 MiB -> dropped clean cache -> headroom 850 MiB (recovered 50 MiB;"
expect_has "  that the wait doubled"                          "next drop allowed in 60 s)"
expect_has "  and that the margin is not held by cache"       "LOW: that drop recovered less than 128 MiB — the margin is held by what the processes themselves have, not by clean file cache; waiting 60 s before the next try"
expect "  a poor attempt is start, attempt, result, LOW"      "$(log_lines)" "4"

scenario 'SEQ_HR="900 1000 700 1100 700 1100"' HELPER_RC=0 READINGS=3 'VALVE_OPTS="MIN_GAP=0"' 
expect "with the wait over, the next drop follows (here on every reading)" "$(count x "$RECLAIM_CALLS")" "3"
expect "  one sudo call each (for a non-root valve)"         "$(count x "$SUDO_CALLS")" "$(( 3 * want_sudo ))"
expect "  recorded"                                           "$(decisions)" "DROPPED->1000 DROPPED->1100 DROPPED->1100 "
expect "  one TRY line ahead of each"                         "$(log_has 'TRY:')" "3"

scenario 'SEQ_HR="800 850 800 850 800 850"' HELPER_RC=0 READINGS=4 'VALVE_OPTS="MIN_GAP=0090 MAX_GAP=0600"' 
expect "a zero-padded MIN_GAP through the loop: one drop, the throttle intact" "$(count x "$RECLAIM_CALLS")" "1"
expect "  then it waits"                                      "$(decisions)" "DROPPED->850 WAIT:throttled WAIT:throttled WAIT:throttled "
expect_has "  and the log names a real wait, not an empty one" "next drop allowed in 180 s)"

scenario 'SEQ_HR="900 900 900 900 900"' HELPER_RC=1 'HELPER_SAYS="helper: device busy"' READINGS=4
expect "a helper that fails: logged with its exit code and its words" "$(log_has 'RECLAIM FAILED at true headroom 900 MiB (exit 1; helper: device busy)')" "1"
expect "  and not retried on the next reading"                "$(count x "$RECLAIM_CALLS")" "1"
expect "  the wait started, and doubled"                      "$(decisions)" "DROP-FAILED WAIT:throttled WAIT:throttled WAIT:throttled "
expect_has "  with what to check"                             "installed, executable and granted in sudoers?"
expect_has "  and the wait after it doubled, as after a poor drop" "Next try in 60 s"

scenario 'SEQ_HR="1500 1500 1500"' READINGS=3
expect "a healthy node: no drop is attempted"                 "$(count x "$RECLAIM_CALLS")" "0"
expect "  and sudo is never asked"                            "$(count x "$SUDO_CALLS")"    "0"
expect "  and the log holds nothing but its start line"       "$(log_lines)" "1"
expect "  every reading is headroom-ok"                       "$(decisions)" "WAIT:headroom-ok WAIT:headroom-ok WAIT:headroom-ok "

scenario 'SEQ_HR="-40 1300 1300"' HELPER_RC=0 READINGS=3
expect "free below the reserve line: it drops at once"        "$(count x "$RECLAIM_CALLS")" "1"
expect_has "  and the log shows the negative margin"           "RECLAIM: true headroom -40 MiB < floor 1000 MiB"

scenario 'SEQ_HR="1500"' READINGS=1 'VALVE_OPTS="FLOOR_MIB=777 MIN_GAP=5 MAX_GAP=50 MIN_DROPPABLE_MIB=9"' 
expect "the start line carries the settings in effect"        "$(log_has 'started (true-headroom floor 777 MiB, interval 0s, min gap 5s, max gap 50s, a drop that recovers less than 9 MiB doubles the wait,')" "1"

scenario 'SEQ_HR="x x x"' READINGS=3
expect "unreadable headroom: the helper is never called"      "$(count x "$RECLAIM_CALLS")" "0"
expect "  nor is sudo"                                        "$(count x "$SUDO_CALLS")"    "0"
expect "  and every reading is recorded as unreadable"        "$(decisions)" "WAIT:unreadable WAIT:unreadable WAIT:unreadable "
expect "  the log says so once, not once per reading"         "$(log_has 'UNREADABLE: the margin cannot be read')" "1"

scenario 'SEQ_HR="1500 x x 1500 1500"' READINGS=5
expect "blind, then able to read again: both said once"       "$(log_has 'UNREADABLE: the margin cannot be read')" "1"
expect_has "  and it says when it can see again"              "READABLE: the margin can be read again — true headroom 1500 MiB"
log_order " UNREADABLE: the margin" " READABLE: the margin"     # anchored: READABLE is inside UNREADABLE
check "  in that order" $? "READABLE is not after UNREADABLE"

part "the CSV keeps an earlier layout readable"
rm -rf "$TMP/state"; mkdir -p "$TMP/state"
printf 'utc,mem_available_mib,decision\n2026-01-01T00:00:00Z,5,OK\n' > "$CSVF"
scenario KEEP_STATE=1 'SEQ_HR="1500"' READINGS=2
expect "a CSV from another layout gets a marker"              "$(grep -c '^# layout change' "$CSVF")" "1"
expect "  and this layout's header right after it"          "$(grep -n -A1 '^# layout change' "$CSVF" | tail -1 | sed -E 's/^[0-9]+[-:]//')" "$CSV_HEADER"
expect "  and its old rows are kept"                          "$(sed -n '2p' "$CSVF")" "2026-01-01T00:00:00Z,5,OK"
scenario KEEP_STATE=1 'SEQ_HR="1500"' READINGS=2
expect "a second start on the converted CSV adds no second marker" "$(grep -c '^# layout change' "$CSVF")" "1"
expect "  and no second header"                               "$(grep -c "^$CSV_HEADER\$" "$CSVF")" "1"
scenario KEEP_STATE=1 'SEQ_HR="1500"' READINGS=2
expect "nor a third"                                         "$(grep -c '^# layout change' "$CSVF")" "1"
: > "$CSVF"
scenario KEEP_STATE=1 'SEQ_HR="1500"' READINGS=2
expect "an empty CSV gets the header, without a marker"      "$(grep -c '^# layout change' "$CSVF")" "0"
expect "  once"                                              "$(grep -c "^$CSV_HEADER\$" "$CSVF")" "1"

# ---- the valve as a program ---------------------------------------------------
part "the valve as a program (it must stop before the loop when it should)"
# Every run as a program points the valve at the fixtures above and at a
# stand-in helper, with `sudo` exported as the recording shim, so even a run
# that should have stopped, but did not, can do nothing but read fixture files
# and append to a file. The alarm ends anything that should have stopped.
prog() {   # prog [KEY=VALUE ... | HELPER=normal|kill] -- [script arguments ...] ; prints the exit code
  local envs=() args=() seen=0 x helper=normal
  for x in "$@"; do
    if [ "$x" = "--" ]; then seen=1
    elif [ "$seen" = 0 ] && { [ "$x" = "HELPER=normal" ] || [ "$x" = "HELPER=kill" ]; }; then helper="${x#HELPER=}"
    elif [ "$seen" = 0 ]; then envs+=("$x")
    else args+=("$x"); fi
  done
  local d="$TMP/prog"; rm -rf "$d"; mkdir -p "$d"
  case "$helper" in
    normal) printf '#!/bin/sh\nprintf "ran\\n" >> "%s/reclaim.calls"\nexit 0\n' "$d" > "$d/fake-reclaim" ;;
    kill)   # Never returns a result: ends the valve itself, as a hang-then-death would.
            printf '#!/bin/sh\nprintf "ran\\n" >> "%s/reclaim.calls"\np="$(ps -o ppid= -p "$PPID" | tr -d " ")"; kill -9 "$p"\n' "$d" > "$d/fake-reclaim" ;;
  esac
  chmod +x "$d/fake-reclaim"
  ( cd "$TMP" && env -u VALVE_LIB_ONLY VALVE_ZONEINFO="$ZONE" VALVE_MEMINFO="$MEM" STATE_DIR="$d/state" RECLAIM="$d/fake-reclaim" HOME="$d" \
      ${envs[@]+"${envs[@]}"} perl -e 'alarm shift; exec @ARGV' 6 bash "${PROG_VALVE:-$VALVE}" ${args[@]+"${args[@]}"} \
      > "$d/out" 2> "$d/err"; echo $? > "$d/rc" )
  cat "$d/rc"
}
refuses_start() {   # true only when the valve exits 2 AND names the setting AND did nothing else
  local rc; rc="$(prog "$@")"
  [ "$rc" = 2 ] && grep -q "mem-floor-valve: .*must be" "$TMP/prog/err" \
    && [ ! -e "$TMP/prog/state" ] && [ ! -f "$TMP/prog/reclaim.calls" ]
}
for kv in "FLOOR_MIB=abc" "INTERVAL=0" "MIN_GAP=0" "MAX_GAP=0" "MIN_DROPPABLE_MIB=-5" "MAX_GAP=10 MIN_GAP=30" \
          "FLOOR_MIB=9999999999" "MIN_GAP=9999999999"; do
  read -r -a parts <<<"$kv"
  refuses_start "${parts[@]}"
  check "settings $kv: refuses to start, naming the setting, writing nothing, calling nothing" $? \
    "exit $(cat "$TMP/prog/rc"); stderr: $(head -c 160 "$TMP/prog/err")"
done
PROG_VALVE="$TMP/broken-valve.sh"; { printf 'if then\n'; cat "$VALVE"; } > "$PROG_VALVE"
refuses_start; rc=$?
[ "$rc" -ne 0 ]
check "'refuses to start' does not accept a valve that merely cannot run (a syntax error also exits 2)" $? "a valve with a syntax error passed for a refusal"
PROG_VALVE=""

rc="$(prog -- --status)"; expect "--status: exit 0"                       "$rc" "0"
expect "  prints exactly the reading and the decision"        "$(cat "$TMP/prog/out")" "true headroom 1300 MiB (floor 1000); clean file pages 940 MiB, mapped in total 1690 MiB, shared memory 820 MiB -> WAIT:headroom-ok"
[ ! -e "$TMP/prog/state" ]; check "  and writes nothing (no state directory)" $? "a state directory was created"
rc="$(prog FLOOR_MIB=2000 -- --status)"
expect "  the same reading against a higher floor would drop"  "$(cat "$TMP/prog/out")" "true headroom 1300 MiB (floor 2000); clean file pages 940 MiB, mapped in total 1690 MiB, shared memory 820 MiB -> DROP"
printf '%s\n' 'MemTotal: 1 kB' > "$TMP/procless"
rc="$(prog VALVE_ZONEINFO="$TMP/procless" VALVE_MEMINFO="$TMP/procless" -- --status)"
expect "  with nothing to read it says UNREADABLE, in the same format, and decides nothing" "$(cat "$TMP/prog/out")" "true headroom UNREADABLE MiB (floor 1000); clean file pages UNREADABLE MiB, mapped in total UNREADABLE MiB, shared memory UNREADABLE MiB -> WAIT:unreadable"

rc="$(prog -- --unknown)";                 expect "an unknown argument: usage, exit 2"                    "$rc" "2"
rc="$(prog -- --status extra)";            expect "an extra argument after --status: usage, exit 2"     "$rc" "2"
rc="$(prog VALVE_LIB_ONLY=1 --)";          expect "executed (not sourced) with VALVE_LIB_ONLY=1: exits 0" "$rc" "0"
[ ! -e "$TMP/prog/state" ];                check "  without starting the loop"        $? "a state directory was created"
[ ! -f "$TMP/prog/reclaim.calls" ];        check "  and without calling the helper"  $? "the helper was called"

rc="$(prog --)";                           expect "with nothing else to do it starts its loop (ended here by the alarm)" "$rc" "142"
grep -q "^$(printf '%s' "$CSV_HEADER" | sed 's/[][\.*^$/]/\\&/g')\$" "$TMP/prog/state/mem-floor.csv"
check "  and takes readings, into a CSV with this layout's header" $? "header: $(head -n 1 "$TMP/prog/state/mem-floor.csv")"
[ "$(grep -vc '^utc,\|^# ' "$TMP/prog/state/mem-floor.csv")" -ge 1 ]; check "  at least one reading was written" $? "no data rows"
[ ! -f "$TMP/prog/reclaim.calls" ];        check "  and never reaches a drop on a healthy node, so sudo is never asked" $? "the helper was called"

# A helper that never returns: the attempt is on record before the act, so the
# log shows what the valve was doing when it stopped. (The stand-in needs ps
# to find the valve's process; without ps the case is skipped, not failed.)
if command -v ps >/dev/null 2>&1; then
  rc="$(prog HELPER=kill FLOOR_MIB=2000 MIN_GAP=0090 MAX_GAP=0600 --)"; expect "a helper that never returns: the valve is ended (here by the stand-in)" "$rc" "137"
  grep -q "started (true-headroom floor 2000 MiB, interval 10s, min gap 90s, max gap 600s," "$TMP/prog/state/mem-floor.log"
  check "  and its start line shows zero-padded settings as plain numbers" $? "start line: $(head -n 1 "$TMP/prog/state/mem-floor.log" 2>/dev/null | cut -c1-110)"
  [ -f "$TMP/prog/state/mem-floor.log" ] && grep -q "TRY: true headroom 1300 MiB < floor 2000 MiB; dropping clean cache" "$TMP/prog/state/mem-floor.log"
  try_ok=$?; check "  and its attempt is on record" $try_ok "no TRY line: $(cat "$TMP/prog/state/mem-floor.log" 2>/dev/null | cut -c1-120)"
  if [ "$try_ok" -ne 0 ]; then
    # Without the attempt on record, "no result" would pass on a missing log.
    check "  and no result is claimed for it" 1 "cannot say: the attempt itself is not on record"
  elif grep -q "RECLAIM" "$TMP/prog/state/mem-floor.log"; then
    check "  and no result is claimed for it" 1 "a RECLAIM line was written"
  else
    ok "  and no result is claimed for it"
  fi
  [ "$(count x "$TMP/prog/reclaim.calls")" = 1 ]; check "  the helper was asked exactly once" $? "calls: $(count x "$TMP/prog/reclaim.calls")"
  if [ "$want_sudo" = 1 ]; then
    [ "$(count x "$SUDO_CALLS")" -ge 1 ]; check "  through the sudo shim, never the real one" $? "the shim recorded nothing"
  else
    ok "  (as root the helper is run directly, by design)"
  fi
else
  part "(skipped: the helper-that-never-returns case needs ps)"
fi

if [ "$(id -u)" -ne 0 ]; then
  d="$TMP/locked"; rm -rf "$d"; mkdir -p "$d"; chmod 555 "$d"
  rc="$(prog STATE_DIR="$d" --)"
  [ "$rc" != 0 ] && [ "$rc" != 142 ] && grep -q "not writable" "$TMP/prog/err"
  check "a state directory it cannot write to: it stops before the loop, and says so" $? "exit $rc; stderr: $(head -c 160 "$TMP/prog/err")"
  [ ! -f "$TMP/prog/reclaim.calls" ]; check "  so nothing is dropped without a record" $? "the helper was called"
  chmod 755 "$d"

  d="$TMP/readonly-log"; rm -rf "$d"; mkdir -p "$d"; : > "$d/mem-floor.log"; chmod 444 "$d/mem-floor.log"
  rc="$(prog FLOOR_MIB=2000 STATE_DIR="$d" --)"
  [ "$rc" != 0 ] && [ "$rc" != 142 ] && grep -q "cannot write its log" "$TMP/prog/err"
  check "a log it cannot append to (mode 444 in a writable directory): it stops before any drop" $? "exit $rc; stderr: $(head -c 160 "$TMP/prog/err")"
  [ ! -f "$TMP/prog/reclaim.calls" ]; check "  and the helper is never called" $? "the helper was called"

  # The CSV already carries this layout's header: nothing needs writing to it
  # at start, so only the permission check can stop the valve here.
  # The CSV becomes unwritable AFTER the start checks (the reading stub does it
  # on its first call): the valve must say so once and keep recording drops in
  # its log.
  : > "$TMP/csv-locked"; rm -f "$TMP/csv-locked.done"
  scenario 'SEQ_HR="800 850 800 850"' HELPER_RC=0 READINGS=4 NO_CSV_ROWS=1 "LOOP_ERR=$TMP/loop.err" </dev/null
  rm -f "$TMP/csv-locked" "$TMP/csv-locked.done"; chmod 755 "$TMP/state" 2>/dev/null
  expect "a CSV that becomes unwritable: the drop still happens" "$(count x "$RECLAIM_CALLS")" "1"
  expect "  and is recorded in the log"                       "$(log_has 'RECLAIM: true headroom 800 MiB')" "1"
  expect "  the CSV failure is said once, not per reading"    "$(log_has 'CSV: a reading could not be appended')" "1"
  if [ -s "$TMP/loop.err" ]; then
    [ "$(grep -c 'Permission denied' "$TMP/loop.err")" = 0 ]; check "  and only there: nothing on stderr per reading" $? "stderr: $(head -c 160 "$TMP/loop.err")"
  else
    ok "  and only there: nothing on stderr per reading"
  fi

  d="$TMP/readonly-csv"; rm -rf "$d"; mkdir -p "$d"; printf '%s\n' "$CSV_HEADER" > "$d/mem-floor.csv"; chmod 444 "$d/mem-floor.csv"
  rc="$(prog FLOOR_MIB=2000 STATE_DIR="$d" --)"
  [ "$rc" != 0 ] && [ "$rc" != 142 ] && grep -q "cannot write its CSV" "$TMP/prog/err"
  check "a CSV it cannot append to: it stops before any drop" $? "exit $rc; stderr: $(head -c 160 "$TMP/prog/err")"
  [ ! -f "$TMP/prog/reclaim.calls" ]; check "  and the helper is never called" $? "the helper was called"
else
  part "(skipped as root: the unwritable state directory, read-only log and read-only CSV cases, since root can write anywhere)"
fi

printf '\ntest-mem-floor-valve: %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ] || exit 1
