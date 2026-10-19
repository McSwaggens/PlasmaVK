#!/bin/bash
# Serializes GPU/CPU-sensitive work on this Mac across agents, sessions and worktrees.
#
#   gpulock.sh excl   CMD...   benchmarks (micro or macro): runs alone. Waits for every shared/cts
#                              holder to finish, blocks new ones while waiting (writer preference),
#                              and waits for outside deqp-vk / engine program* / harness locks.
#   gpulock.sh shared CMD...   builds and tests (make, clang, make test/check, smoke tests): any number
#                              at once, never during an excl holder.
#   gpulock.sh cts    CMD...   a deqp-vk run: shared, plus only one cts holder at a time.
#   gpulock.sh status          who holds / waits.
#
# GPULOCK_WAIT=secs (default 540) bounds the wait: on timeout it exits 75 without running CMD,
# so a caller limited to 10-minute tool calls can retry. CMD's exit status is passed through.
# Every run is logged to $L/log (start, end, mode, seconds, cwd, command).
L=${GPULOCK_DIR:-$HOME/plasmavk-bench/.gpulock}
mkdir -p "$L/shared" "$L/want"
WAIT=${GPULOCK_WAIT:-540}
mode=${1:-}; shift || true

alive() { local p; for p in $1; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }
clean() {  # drop entries whose processes are gone
	local f
	for f in "$L"/shared/* "$L"/want/*; do
		[ -e "$f" ] || continue
		alive "$(cat "$f" 2>/dev/null)" || rm -f "$f"
	done
	for d in "$L/excl" "$L/cts"; do
		[ -d "$d" ] || continue
		local p; p=$(cat "$d/pid" 2>/dev/null)
		if [ -z "$p" ]; then  # being created: give it a moment
			[ "$(($(date +%s) - $(stat -f %m "$d")))" -gt 10 ] && rm -rf "$d"
		elif ! alive "$p"; then rm -rf "$d"; fi
	done
}
nlive() { local n=0 f; for f in "$1"/*; do [ -e "$f" ] && n=$((n + 1)); done; echo $n; }
describe() {
	local f d
	for d in "$L/excl" "$L/cts"; do
		[ -d "$d" ] && echo "  $(basename "$d"): $(cat "$d/info" 2>/dev/null)"
	done
	for f in "$L"/shared/* "$L"/want/*; do
		[ -e "$f" ] && echo "  $(basename "$(dirname "$f")"): $(sed -n 2p "$f")"
	done
}
outside() {  # GPU work not under this lock
	local r=""
	pgrep -x deqp-vk >/dev/null && [ ! -d "$L/cts" ] && r="$r deqp-vk"
	ps -Ao pid=,comm= | awk '$2 ~ /\/program[^\/]*$/' | grep -q . && r="$r engine-program"
	[ -d "$HOME/plasmavk-bench/.bench.lock" ] && r="$r harness-run.sh"
	echo "$r"
}
log() { echo "$(date '+%F %T') $*" >>"$L/log"; }

info="pid $$ cwd $PWD: $*"
case "$mode" in
status) clean; describe; echo "outside:$(outside)"; tail -5 "$L/log" 2>/dev/null; exit 0 ;;
excl | shared | cts) ;;
*) sed -n 2,16p "$0"; exit 2 ;;
esac
[ $# -gt 0 ] || { echo "gpulock: no command" >&2; exit 2; }

t0=$(date +%s); said=0
waited() {
	local now; now=$(date +%s)
	if [ $((now - t0)) -ge "$WAIT" ]; then
		echo "gpulock: gave up after ${WAIT}s waiting for:" >&2; describe >&2
		echo "  outside:$(outside)" >&2
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
	rm -f "$L/want/$$" "$L/shared/$$"
	[ -n "$me" ] && [ "$(cat "$me/pid" 2>/dev/null | cut -d' ' -f1)" = "$$" ] && rm -rf "$me"
	[ "$(cat "$L/cts/pid" 2>/dev/null | cut -d' ' -f1)" = "$$" ] && rm -rf "$L/cts"
}
trap 'cleanup; exit 130' INT TERM HUP
trap cleanup EXIT

if [ "$mode" = excl ]; then
	printf '%s\n%s\n' $$ "$info" >"$L/want/$$"
	while :; do
		clean
		if mkdir "$L/excl" 2>/dev/null; then
			echo "$info" >"$L/excl/info"; echo $$ >"$L/excl/pid"; me=$L/excl; break
		fi
		waited
	done
	while [ "$(nlive "$L/shared")" -gt 0 ] || [ -d "$L/cts" ] || [ -n "$(outside)" ]; do clean; waited; done
	rm -f "$L/want/$$"
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
