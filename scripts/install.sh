#!/bin/sh
# RockNPU one-step installer for RK3588 boards.
#
# Builds libggml-rocknpu.so (the RockNPU backend for llama.cpp / Ollama) and
# writes an environment file that makes llama.cpp-family frontends use it.
#
#   ./scripts/install.sh                 # backend only (you already have llama.cpp / Ollama)
#   ./scripts/install.sh --with-llama    # also build llama.cpp's llama-server / llama-cli
#
# Options (environment variables):
#   PREFIX     install directory            (default: ~/.local/share/rocknpu)
#   LLAMA_REF  llama.cpp tag/commit to use  (default: b10969, the revision Ollama 0.34.x pins)
set -eu

PREFIX=${PREFIX:-"$HOME/.local/share/rocknpu"}
LLAMA_REF=${LLAMA_REF:-b10969}
WITH_LLAMA=0
for arg in "$@"; do
    case "$arg" in
        --with-llama) WITH_LLAMA=1 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

ROOT=$(cd "$(dirname "$0")/.." && pwd)
say() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# 1. Hardware / permission checks ------------------------------------------
[ "$(uname -m)" = aarch64 ] || die "RockNPU runs on RK3588 (aarch64) boards."
if [ ! -e /dev/accel/accel0 ]; then
    die "/dev/accel/accel0 not found. Use a kernel with the 'rocket' NPU driver
       (Linux 6.18+ / Armbian 'current'), then reboot."
fi
if [ ! -r /dev/accel/accel0 ] || [ ! -w /dev/accel/accel0 ]; then
    warn "no access to /dev/accel/accel0. Run once:"
    warn "    sudo usermod -aG render \"$USER\"   (then log out and back in)"
fi

# 2. Build tools --------------------------------------------------------------
missing=""
for tool in git cmake c++ curl; do
    command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"
done
if [ -n "$missing" ]; then
    say "installing build tools:$missing"
    sudo apt-get update && sudo apt-get install -y git cmake build-essential curl
fi
if ! command -v cargo >/dev/null 2>&1; then
    [ -x "$HOME/.cargo/bin/cargo" ] || {
        say "installing the Rust toolchain (rustup)"
        curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal
    }
    PATH="$HOME/.cargo/bin:$PATH"
fi

# 3. llama.cpp sources (headers for the backend ABI, optional binaries) ------
SRC="$PREFIX/src/llama.cpp"
mkdir -p "$PREFIX/src" "$PREFIX/lib"
if [ ! -d "$SRC/.git" ]; then
    say "fetching llama.cpp $LLAMA_REF"
    git clone --depth 1 --branch "$LLAMA_REF" https://github.com/ggml-org/llama.cpp "$SRC" 2>/dev/null ||
        { git clone https://github.com/ggml-org/llama.cpp "$SRC" && git -C "$SRC" checkout "$LLAMA_REF"; }
fi

# 4. RockNPU backend ------------------------------------------------------------
say "building the RockNPU backend (first build takes a few minutes)"
cmake -S "$ROOT/adapters/ggml-rocknpu" -B "$PREFIX/build/ggml-rocknpu" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_SOURCE_DIR="$SRC/ggml" >/dev/null
cmake --build "$PREFIX/build/ggml-rocknpu" -j"$(nproc)"
cp "$PREFIX/build/ggml-rocknpu/libggml-rocknpu.so" "$PREFIX/lib/"

if [ "$WITH_LLAMA" = 1 ]; then
    say "building llama.cpp ($LLAMA_REF)"
    cmake -S "$SRC" -B "$PREFIX/build/llama.cpp" -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF >/dev/null
    cmake --build "$PREFIX/build/llama.cpp" -j"$(nproc)" --target llama-server llama-cli
fi

# 5. Environment file ------------------------------------------------------------
ENV="$PREFIX/rocknpu.env"
cat >"$ENV" <<EOF
# RockNPU: load with   . $ENV
export GGML_BACKEND_PATH="$PREFIX/lib/libggml-rocknpu.so"
# route llama.cpp / Ollama work to the NPU backend
export LLAMA_ARG_DEVICE=ROCKNPU0
# keep GGUF weights in their original layout so the NPU can read them
export LLAMA_ARG_REPACK=false
# shorter OpenMP spin: the CPU threads otherwise compete with NPU host work
export GOMP_SPINCOUNT=20000
EOF
if [ "$WITH_LLAMA" = 1 ]; then
    echo "export PATH=\"$PREFIX/build/llama.cpp/bin:\$PATH\"" >>"$ENV"
fi

say "done."
cat <<EOF

  Load the settings in every new shell (or add this line to ~/.bashrc):

      . $ENV

  Check that the NPU is visible:

      llama-server --list-devices      # should list: ROCKNPU0: RockNPU RK3588

  Optional one-time speed tuning (IRQ routing, CPU idle):  sudo $ROOT/scripts/rocknpu-tune.sh install
EOF
