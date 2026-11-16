#!/usr/bin/env bash
# RU: Контракт CLI на изолированной границе Docker, без контейнеров и сети.
# EN: CLI contracts at an isolated Docker boundary, without containers or network.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export HOME="$TMP/home" DOCKER_CONFIG="$TMP/config" STATE="$TMP/state"
# RU: Даже при потере подмены Docker не достигнет демона.
# EN: Even if fixture resolution fails, Docker cannot reach a daemon.
unset DOCKER_CONTEXT DOCKER_TLS_VERIFY DOCKER_CERT_PATH
export DOCKER_HOST="unix://$TMP/no-daemon"
mkdir -p "$TMP/bin" "$TMP/fallback" "$TMP/work" "$HOME" "$DOCKER_CONFIG" "$STATE"
ln -s "$REPO/docker" "$TMP/work/docker"
export PATH="$TMP/bin:$TMP/fallback:$PATH" GPU=cpu HEADLESS=1
unset VESTA_IMAGE DOCKER_EXTRA NAME
export VESTA_MODELS="$HOME/models"
mkdir -p "$VESTA_MODELS"
cd "$TMP/work"
cp "$REPO/docker/test/fixtures/docker" "$TMP/bin/docker"
printf '#!/usr/bin/env bash\nexit 0\n' > "$TMP/bin/xhost"
# RU: Ловушка для безопасного опровержения проверки. EN: Tripwire for safe guard falsification.
printf '#!/usr/bin/env bash\n: > "$STATE/escaped"\nexit 112\n' > "$TMP/fallback/docker"
chmod +x "$TMP/bin/docker" "$TMP/bin/xhost" "$TMP/fallback/docker"
DEFAULT=ghcr.io/ssskiv/vesta-sim:nav2ai
DIGEST=ghcr.io/ssskiv/vesta-sim@sha256:cf0a98f3e5544443acee5c4e0e95dfdd6349c9d653b931589762435bc56c6524
RUN="$REPO/ros2_ws/src/offroad_nav2/docker/run.sh"
fail() { echo "FAIL: $*" >&2; cat "$STATE/output" >&2; exit 1; }
equal() { [ "$1" = "$2" ] || fail "expected [$2], got [$1]"; }
absent() { [ ! -e "$STATE/$1" ] || fail "unexpected effect: $1"; }
reset() {
    rm -rf "$STATE"
    mkdir "$STATE"
    : > "$STATE/output"
    unset VESTA_IMAGE
}
cache() { echo "$1" > "$STATE/ref"; echo "${2:-linux/amd64 2}" > "$STATE/meta"; }
fixture_ready() {
    hash -r
    [ -z "${DOCKER_CONTEXT:-}" ] && [ "$DOCKER_CONFIG" = "$TMP/config" ] && [ "$DOCKER_HOST" = "unix://$TMP/no-daemon" ] &&
        [ -x "$TMP/bin/docker" ] && [ "$(command -v docker)" = "$TMP/bin/docker" ] &&
        [ -x "$TMP/bin/xhost" ] && [ "$(command -v xhost)" = "$TMP/bin/xhost" ]
}
invoke() {
    local expected="$1" rc=0
    shift
    fixture_ready || { echo 'FAIL: unsafe Docker fixture' >&2; exit 111; }
    "$@" > "$STATE/output" 2>&1 || rc=$?
    if [ "$expected" = fail ]; then [ "$rc" != 0 ] || fail 'unexpected success'
    else equal "$rc" "$expected"; fi
}
make_cli() { make --no-print-directory -C "$TMP/work" -f "$REPO/Makefile" "$@"; }
acquire() { bash -c 'source "$1/docker/image.sh"; vesta_image_acquire "${2:-}"' _ "$REPO" "${1:-}"; }
pass() { echo "PASS: $*"; }

# RU: Поломка подмены останавливает тест, а не запускает настоящий Docker.
# EN: A broken fixture stops the test instead of invoking real Docker.
for fault in mode resolution; do
    reset; rc=0; saved_path="$PATH"
    if [ "$fault" = mode ]; then chmod -x "$TMP/bin/docker"; else PATH="${PATH#*:}"; fi
    (invoke 0 make_cli image) > "$TMP/fixture-failure.log" 2>&1 || rc=$?
    chmod +x "$TMP/bin/docker"; PATH="$saved_path"; hash -r
    equal "$rc" 111; absent calls; absent escaped; absent built; absent ran
    pass "fixture $fault failure exits 111 before entrypoint invocation"
done
reset; make_path="$(command -v make)"; mkdir "$TMP/no-client"
invoke 2 env PATH="$TMP/no-client" "$make_path" --no-print-directory -C "$TMP/work" -f "$REPO/Makefile" image
absent calls; absent built; absent ran; pass 'intentional missing Docker client retains Make failure'

# Helper owner: defaults, export, cache, refresh, and validation failures.
for selection in unset empty; do
    reset
    [ "$selection" != empty ] || export VESTA_IMAGE=
    invoke 0 bash -c 'source "$1/docker/image.sh"; bash -c '\''printf "%s" "$VESTA_IMAGE"'\''' _ "$REPO"
    equal "$(<"$STATE/output")" "$DEFAULT"
    pass "$selection selection exports default"
done
for ref in "$DEFAULT" "$DIGEST"; do
    reset; export VESTA_IMAGE="$ref"
    invoke 0 acquire
    equal "$(<"$STATE/ref")" "$ref"
    equal "$(<"$STATE/pulls")" "$ref"
    pass "missing registry image acquired: $ref"
done
reset; cache "$DEFAULT"; echo 23 > "$STATE/pull-fail"
invoke 0 acquire
absent pulls
pass 'compatible cache works offline'

for metadata in 'linux/amd64 1' 'linux/arm64 2' 'windows/amd64 2'; do
    for origin in cached pulled; do
        reset
        if [ "$origin" = cached ]; then cache "$DEFAULT" "$metadata"
        else echo "$metadata" > "$STATE/remote-meta"; fi
        invoke fail acquire
        absent built; absent ran
        pass "$origin incompatible image rejected: $metadata"
    done
done
reset; export VESTA_IMAGE=vesta-sim:latest
invoke fail acquire
absent pulls; absent built
grep -q 'make image-build' "$STATE/output" || fail 'missing local build guidance'
pass 'missing local image refuses registry acquisition'
reset; cache vesta-sim:latest; export VESTA_IMAGE=vesta-sim:latest
invoke 0 acquire
absent pulls
pass 'compatible explicit local image accepted'
for ref in '--help' 'bad ref' $'bad\nref' 'bad"quote' '$(touch SHOULD_NOT_EXIST)' 'bad;true'; do
    reset; export VESTA_IMAGE="$ref"
    invoke fail acquire
    absent built; absent ran
    [ ! -e SHOULD_NOT_EXIST ] || fail 'image override executed shell syntax'
    pass "malformed reference fails safely: $(printf %q "$ref")"
done

# Wiring: Make's acquisition, refresh, local builds, and runtime launch.
reset
invoke 0 make_cli image
equal "$(<"$STATE/ref")" "$DEFAULT"; absent built; absent ran
pass 'make image acquires public default without build'
reset; cache "$DIGEST"; export VESTA_IMAGE="$DIGEST"
invoke 0 make_cli image "VESTA_IMAGE=$DIGEST"
absent pulls; absent built
pass 'make image honors command line digest override offline'
reset; cache "$DEFAULT" 'linux/amd64 1'
invoke 0 make_cli pull
equal "$(<"$STATE/meta")" 'linux/amd64 2'
equal "$(<"$STATE/pulls")" "$DEFAULT"; absent built
pass 'make pull refreshes stale cache and validates replacement'
for status in 23 130; do
    for entry in helper make run; do
        reset; echo "$status" > "$STATE/pull-fail"
        case "$entry" in
            helper) invoke "$status" acquire ;;
            make) invoke fail make_cli image ;;
            run) invoke "$status" "$RUN" 'echo should-not-run' ;;
        esac
        absent ref; absent built; absent ran
        rm "$STATE/pull-fail"
        invoke 0 acquire
        equal "$(<"$STATE/ref")" "$DEFAULT"
        pass "$entry failed/interrupted pull ($status) stops; retry acquires"
    done
done
reset; cache "$DEFAULT"; echo 23 > "$STATE/pull-fail"
invoke fail make_cli pull
absent built; absent ran
pass 'explicit refresh failure fails even with compatible cache'
reset; touch "$STATE/pull-lie"
invoke fail make_cli image
absent ref; absent built; absent ran
pass 'successful pull exit without an inspectable image is rejected'
for entry in image run; do
    reset; cache "$DEFAULT" 'linux/amd64 1'
    if [ "$entry" = image ]; then invoke fail make_cli image
    else invoke fail "$RUN" 'echo should-not-run'; fi
    absent pulls; absent built; absent ran
    pass "$entry stale cached marker stops without substitution"
done
reset; touch "$STATE/build-fail"
invoke fail make_cli rebuild
absent built; absent ran
pass 'rebuild failure stops before cleanup or success'
for target in image-build rebuild; do
    reset; export VESTA_IMAGE="$DIGEST"
    mkdir -p "$TMP/work/.build"
    invoke 0 make_cli "$target"
    equal "$(<"$STATE/built")" vesta-sim:latest
    absent pulls
    if [ "$target" = rebuild ]; then
        equal "$(<"$STATE/build-options")" $'--no-cache\nsim'
        equal "$(<"$STATE/ran")" vesta-sim:latest
    else equal "$(<"$STATE/build-options")" sim; fi
    pass "$target builds local tag despite runtime digest override"
done
for target in up shell onnx; do
    reset; cache "$DIGEST"; export VESTA_IMAGE="$DIGEST"
    invoke 0 make_cli "$target" 'CMD=printf "command with spaces"'
    equal "$(<"$STATE/ran")" "$DIGEST"; absent pulls; absent built
    pass "make $target uses acquired image with implicit pulls/builds disabled"
done
reset; export VESTA_IMAGE="$DIGEST" NAME='nav_contract' DOCKER_EXTRA='-e CONTRACT=selected'
invoke 0 "$RUN" 'printf "%s\n" "two words"; exit 7'
equal "$(<"$STATE/ran")" "$DIGEST"; absent built
equal "$(<"$STATE/command")" $'bash\n-lc\nsource /opt/ros/humble/setup.bash; [ -f /ws/install/setup.bash ] && source /ws/install/setup.bash; printf "%s\\n" "two words"; exit 7'
for arg in nav_contract CONTRACT=selected "$HOME/.cache/vesta-nav:/ws" "$REPO:/vesta" "$VESTA_MODELS:/models:ro"; do
    grep -Fxq -- "$arg" "$STATE/runtime-options" || fail "lost runtime argument: $arg"
done
pass 'run.sh preserves image command name environment and cache mounts'
reset; cache "$DEFAULT"
invoke 0 env -u VESTA_MODELS "$RUN" 'true'
grep -Fxq -- "$REPO/ml_models:/models:ro" "$STATE/runtime-options" || fail 'models must default to the checkout ml_models directory'
pass 'run.sh defaults models to the checkout regardless of HOME or working directory'
reset; cache "$DEFAULT"; export RUN_EXIT=7
invoke 7 "$RUN" 'exit 7'
pass 'run.sh propagates Docker runtime failure'

# Smoke CLI owner: reject invalid input before dependency execution.
SMOKE="$REPO/docker/smoke-image.sh"
unset RUN_EXIT
for input in missing empty extra; do
    reset
    case "$input" in
        missing) invoke 2 bash "$SMOKE" ;;
        empty) invoke 2 bash "$SMOKE" '' ;;
        extra) invoke 2 bash "$SMOKE" "$DIGEST" extra ;;
    esac
    absent calls
    pass "smoke $input argument rejected before Docker"
done
reset
invoke 125 bash "$SMOKE" --help
absent ran; absent pulls; absent built
pass 'smoke leading-option image fails instead of successful Docker help'
reset
invoke 125 bash "$SMOKE" vesta-smoke-missing:never
absent ran; absent pulls; absent built
pass 'smoke missing image fails without acquisition'
reset; cache "$DIGEST"
invoke 0 bash "$SMOKE" "$DIGEST"
equal "$(<"$STATE/ran")" "$DIGEST"; absent pulls; absent built
pass 'smoke explicit cached image reaches runtime without acquisition'
reset; cache "$DIGEST"; export RUN_EXIT=7
invoke 7 bash "$SMOKE" "$DIGEST"
pass 'smoke propagates Docker runtime failure'
echo 'All image contracts passed.'
