#!/bin/bash
# Serializes GPU/CPU-sensitive work on this Mac across agents, sessions, worktrees and codebases (PlasmaVK, the engine).
#
#   gpulock.sh excl   CMD...   benchmarks, profiles, anything timed or GPU-heavy: runs alone. Exclusive runs queue
#                              first come, first served (a loop of them goes behind those already waiting); new
#                              shared/cts holders wait while one is queued (writer preference), and it waits for
#                              outside deqp-vk / engine program* / harness locks.
#   gpulock.sh shared CMD...   builds and tests (make, clang, make test/check, smoke tests): any number at once, never
#                              during an excl holder.
#   gpulock.sh cts    CMD...   a deqp-vk run: shared, plus only one cts holder at a time.
#   gpulock.sh status          who holds / waits.
#
# GPULOCK_WAIT=secs (default 540) bounds the wait: on timeout it exits 75 without running CMD, so a caller limited to
# 10-minute tool calls can retry; an excl caller keeps its place in the queue for GPULOCK_HOLD secs (default 120) if
# it retries the same command from the same directory. CMD's exit status is passed through. Every run is logged to
# $L/log (start, end, mode, seconds, cwd, command). Installed as ~/bin/gpulock; the lock is ~/.gpulock.
L=${GPULOCK_DIR:-$HOME/.gpulock}
mkdir -p "$L/shared" "$L/want"
WAIT=${GPULOCK_WAIT:-540}
HOLD=${GPULOCK_HOLD:-120}
mode=${1:-}; shift || true

alive() { local p; for p in $1; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }
clean() {  # drop entries whose processes are gone, and expired places in the queue
	local f p now
	now=$(date +%s)
	for f in "$L"/shared/* "$L"/want/*; do
		[ -e "$f" ] || continue
		p=$(sed -n 1p "$f" 2>/dev/null)
		case $p in
		hold:*) [ "${p#hold:}" -gt "$now" ] 2>/dev/null || rm -f "$f" ;;
		*) alive "$p" || rm -f "$f" ;;
		esac
	done
	for d in "$L/excl" "$L/cts"; do
		[ -d "$d" ] || continue
		p=$(cat "$d/pid" 2>/dev/null)
		if [ -z "$p" ]; then  # being created: give it a moment
			[ "$((now - $(stat -f %m "$d")))" -gt 10 ] && rm -rf "$d"
		elif ! alive "$p"; then rm -rf "$d"; fi
	done
}
nlive() { local n=0 f; for f in "$1"/*; do [ -e "$f" ] && n=$((n + 1)); done; echo $n; }
describe() {
	local f d
	for d in "$L/excl" "$L/cts"; do
		[ -d "$d" ] && echo "  $(basename "$d"): $(cat "$d/info" 2>/dev/null)"
	done
	for f in "$L"/shared/*; do
		[ -e "$f" ] && echo "  shared: $(sed -n 2p "$f")"
	done
	for f in $(ls "$L/want" 2>/dev/null | sort); do  # (in queue order)
		case $(sed -n 1p "$L/want/$f") in
		hold:*) echo "  want (place kept): $(sed -n 2p "$L/want/$f")" ;;
		*) echo "  want: $(sed -n 2p "$L/want/$f")" ;;
		esac
	done
}
outside() {  # GPU work not under this lock
	local r=""
	[ "${GPULOCK_OUTSIDE:-1}" = 0 ] && return  # (testing the lock itself)
	pgrep -x deqp-vk >/dev/null && [ ! -d "$L/cts" ] && r="$r deqp-vk"
	ps -Ao pid=,comm= | awk '$2 ~ /\/program[^\/]*$/' | grep -q . && r="$r engine-program"
	[ -d "$HOME/plasmavk-bench/.bench.lock" ] && r="$r harness-run.sh"
	echo "$r"
}
log() { echo "$(date '+%F %T') $*" >>"$L/log"; }
ticket() {  # the name of a new place at the back of the excl queue (names sort in arrival order)
	local n=0 s
	while ! mkdir "$L/seq.lock" 2>/dev/null; do
		n=$((n + 1))
		[ $n -gt 50 ] && { rm -rf "$L/seq.lock"; n=0; }  # (left by a killed caller)
		sleep 0.1
	done
	s=$(($(cat "$L/seq" 2>/dev/null || echo 0) + 1))
	echo $s >"$L/seq"
	rmdir "$L/seq.lock"
	printf 's%012d-%d' $s $$
}

key="cwd $PWD: $*"
info="pid $$ $key"
case "$mode" in
status) clean; describe; echo "outside:$(outside)"; tail -5 "$L/log" 2>/dev/null; exit 0 ;;
excl | shared | cts) ;;
*) sed -n 2,16p "$0"; exit 2 ;;
esac
[ $# -gt 0 ] || { echo "gpulock: no command" >&2; exit 2; }

t0=$(date +%s); said=0
want=""
waited() {
	local now; now=$(date +%s)
	if [ $((now - t0)) -ge "$WAIT" ]; then
		echo "gpulock: gave up after ${WAIT}s waiting for:" >&2; describe >&2
		echo "  outside:$(outside)" >&2
		if [ -n "$want" ]; then  # keep the place for a retry of the same command
			printf 'hold:%s\n%s\n' $((now + HOLD)) "$key" >"$want"
			want=""
		fi
		cleanup; exit 75
	fi
	if [ $said = 0 ] || [ $(((now - t0) % 60)) -lt 2 ]; then
		echo "gpulock: $mode waiting ($((now - t0))s):" >&2; describe >&2; o=$(outside); [ -n "$o" ] && echo "  outside:$o" >&2
		said=1
	fi
	sleep 2
}
me=""
cleanup() {
	[ -n "$want" ] && rm -f "$want"
	rm -f "$L/shared/$$"
	[ -n "$me" ] && [ "$(cat "$me/pid" 2>/dev/null | cut -d' ' -f1)" = "$$" ] && rm -rf "$me"
	[ "$(cat "$L/cts/pid" 2>/dev/null | cut -d' ' -f1)" = "$$" ] && rm -rf "$L/cts"
}
trap 'cleanup; exit 130' INT TERM HUP
trap cleanup EXIT

if [ "$mode" = excl ]; then
	for f in "$L"/want/*; do  # a place this command kept when it gave up
		[ -e "$f" ] || continue
		case $(sed -n 1p "$f" 2>/dev/null) in
		hold:*) [ "$(sed -n 2p "$f")" = "$key" ] && { want=$f; break; } ;;
		esac
	done
	[ -n "$want" ] || want="$L/want/$(ticket)"
	printf '%s\n%s\n' $$ "$info" >"$want"
	while :; do
		clean
		# (want files of callers of an older gpulock.sh, named by a pid alone, keep no place: those callers
		# take the lock whenever it is free, and would otherwise always come first)
		first=$(ls "$L/want" 2>/dev/null | grep '^s' | sort | head -1)
		if [ "$L/want/$first" = "$want" ] && mkdir "$L/excl" 2>/dev/null; then
			echo "$info" >"$L/excl/info"; echo $$ >"$L/excl/pid"; me=$L/excl; break
		fi
		waited
	done
	while [ "$(nlive "$L/shared")" -gt 0 ] || [ -d "$L/cts" ] || [ -n "$(outside)" ]; do clean; waited; done
	rm -f "$want"; want=""
	# Builds outside the lock skew measurements: give them a minute to finish, then say so.
	for _ in $(seq 30); do pgrep -x clang >/dev/null || pgrep -x ld >/dev/null || break; sleep 2; done
	busy=$(pgrep -x clang | wc -l | tr -d ' ')
	[ "$busy" -gt 0 ] && echo "gpulock: WARNING $busy clang processes running outside the lock; numbers may be noisy" >&2
else
	while :; do
		clean
		if [ ! -d "$L/excl" ] && [ "$(nlive "$L/want")" = 0 ]; then
			printf '%s\n%s\n' $$ "$info" >"$L/shared/$$"
			if [ -d "$L/excl" ]; then rm -f "$L/shared/$$"; else
				if [ "$mode" = cts ]; then
					if mkdir "$L/cts" 2>/dev/null; then echo "$info" >"$L/cts/info"; echo $$ >"$L/cts/pid"; break; fi
					rm -f "$L/shared/$$"
				else break; fi
			fi
		fi
		waited
	done
fi

log "start $mode pid $$ cwd $PWD: $*"
s=$(date +%s)
"$@" &
child=$!
# Keep the child's pid with ours so a killed wrapper doesn't free the lock while CMD still runs.
if [ "$mode" = excl ]; then echo "$$ $child" >"$L/excl/pid"
else printf '%s\n%s\n' "$$ $child" "$info" >"$L/shared/$$"; [ "$mode" = cts ] && echo "$$ $child" >"$L/cts/pid"; fi
trap 'kill -TERM $child 2>/dev/null; wait $child; cleanup; exit 130' INT TERM HUP
wait $child; rc=$?
log "end   $mode pid $$ rc $rc $(($(date +%s) - s))s: $*"
exit $rc
