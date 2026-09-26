#!/bin/sh
# System settings that make RockNPU faster. Nothing here changes the kernel
# or any file on disk except the optional systemd unit; `restore` undoes it.
#
#   sudo ./scripts/rocknpu-tune.sh apply     # apply now (until reboot)
#   sudo ./scripts/rocknpu-tune.sh install   # apply now and at every boot (systemd)
#   sudo ./scripts/rocknpu-tune.sh dvfs      # optional: NPU frequency scaling module (see below)
#   sudo ./scripts/rocknpu-tune.sh restore   # back to the distribution defaults
#   ./scripts/rocknpu-tune.sh status         # show the current state
#
# What it does:
#   * NPU interrupts go to the little cores 0-2 (one per NPU core), away from
#     the big cores that run llama.cpp;
#   * the deep "cpu-sleep" idle state is disabled (waking from it costs
#     ~220 us, paid on every NPU job completion);
#   * CPU frequency governor = performance;
#   * if the NPU exposes devfreq, it runs at the highest frequency the
#     current rail voltage allows.
#
# `dvfs` (optional): the mainline rocket driver has no frequency scaling and
# leaves the NPU at its 200 MHz boot clock (prompt processing ~40 % slower,
# NPU decode ~60 % slower than at 700 MHz). It builds the out-of-tree rocket
# driver with devfreq from https://github.com/sky-rk3588/rk3588-npu-gpu
# (pinned commit below) against the running kernel's headers, swaps it in
# and, once `install` has been run, again at every boot. Nothing is written
# to /lib/modules or the boot files; `restore` returns to the stock module at
# the next reboot. No voltage change is made: at the stock 800 mV rail the
# driver allows up to 700 MHz, which is also where LLM throughput saturates.
set -eu

DVFS_REPO=https://github.com/sky-rk3588/rk3588-npu-gpu
DVFS_COMMIT=ed52a89afa8e68fedf636c8e891bd8fc47e82d26
DVFS_KO=/usr/local/lib/rocknpu/rocket-devfreq.ko

mode=${1:-status}
npu_devfreq=/sys/class/devfreq/fdab0000.npu

irq_for() {
    awk -v dev="$1" '$0 ~ dev {gsub(":", "", $1); print $1; exit}' /proc/interrupts
}

status() {
    for dev in fdab0000.npu fdac0000.npu fdad0000.npu; do
        irq=$(irq_for "$dev")
        [ -n "$irq" ] && echo "$dev irq $irq -> cpus $(cat /proc/irq/"$irq"/smp_affinity_list)"
    done
    for d in /sys/devices/system/cpu/cpu*/cpuidle/state*; do
        [ "$(cat "$d/name" 2>/dev/null)" = cpu-sleep ] &&
            echo "$(basename "$(dirname "$(dirname "$d")")") cpu-sleep disabled=$(cat "$d/disable")"
    done
    for p in /sys/devices/system/cpu/cpufreq/policy*; do
        echo "$(basename "$p") governor=$(cat "$p/scaling_governor")"
    done
    if [ -d "$npu_devfreq" ]; then
        echo "npu devfreq governor=$(cat $npu_devfreq/governor) cur=$(cat $npu_devfreq/cur_freq) max=$(cat $npu_devfreq/max_freq)"
    else
        echo "npu devfreq: not available (the NPU stays at its boot clock)"
    fi
}

need_root() {
    [ "$(id -u)" -eq 0 ] || { echo "run as root: sudo $0 $mode" >&2; exit 1; }
}

set_cpu_sleep() {
    for d in /sys/devices/system/cpu/cpu*/cpuidle/state*; do
        [ "$(cat "$d/name" 2>/dev/null)" = cpu-sleep ] && echo "$1" >"$d/disable"
    done
}

swap_dvfs_module() {
    [ -f "$DVFS_KO" ] || return 0
    [ -d "$npu_devfreq" ] && return 0
    if command -v modinfo >/dev/null 2>&1 &&
        ! modinfo -F vermagic "$DVFS_KO" 2>/dev/null | grep -q "^$(uname -r) "; then
        echo "rocket-devfreq.ko was built for another kernel; run: sudo $0 dvfs" >&2
        return 0
    fi
    if command -v fuser >/dev/null 2>&1 && fuser /dev/accel/accel0 >/dev/null 2>&1; then
        echo "the NPU is in use; stop llama.cpp/Ollama first" >&2
        return 0
    fi
    rmmod rocket 2>/dev/null || true
    if ! insmod "$DVFS_KO"; then
        echo "loading rocket-devfreq.ko failed; restoring the stock driver" >&2
        modprobe rocket || true
    fi
}

build_dvfs() {
    need_root
    src=/usr/local/src/rk3588-npu-gpu
    [ -d /lib/modules/"$(uname -r)"/build ] || {
        echo "kernel headers missing (Armbian: sudo armbian-config -> Kernel headers, or apt install linux-headers-current-rockchip64)" >&2
        exit 1
    }
    if [ ! -d "$src/.git" ]; then
        git clone "$DVFS_REPO" "$src"
    fi
    git -C "$src" fetch -q origin || true
    git -C "$src" checkout -q "$DVFS_COMMIT"
    make -C /lib/modules/"$(uname -r)"/build M="$src/npu/driver" modules
    install -D -m 0644 "$src/npu/driver/rocket.ko" "$DVFS_KO"
    swap_dvfs_module
    apply
}

apply() {
    need_root
    swap_dvfs_module
    core=0
    for dev in fdab0000.npu fdac0000.npu fdad0000.npu; do
        irq=$(irq_for "$dev")
        [ -n "$irq" ] && echo "$core" >/proc/irq/"$irq"/smp_affinity_list
        core=$((core + 1))
    done
    set_cpu_sleep 1
    for p in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
        echo performance >"$p"
    done
    if [ -d "$npu_devfreq" ] && grep -qw userspace $npu_devfreq/available_governors; then
        echo userspace >$npu_devfreq/governor
        best=$(tr ' ' '\n' <$npu_devfreq/available_frequencies | sort -n | tail -1)
        echo "$best" >$npu_devfreq/max_freq 2>/dev/null || true
        # the driver refuses frequencies the rail voltage cannot sustain;
        # walk down until one is accepted
        for f in $(tr ' ' '\n' <$npu_devfreq/available_frequencies | sort -rn); do
            echo "$f" >$npu_devfreq/userspace/set_freq 2>/dev/null && [ "$(cat $npu_devfreq/cur_freq)" = "$f" ] && break
        done
    fi
    status
}

restore() {
    need_root
    all="0-$(($(getconf _NPROCESSORS_ONLN) - 1))"
    for dev in fdab0000.npu fdac0000.npu fdad0000.npu; do
        irq=$(irq_for "$dev")
        [ -n "$irq" ] && echo "$all" >/proc/irq/"$irq"/smp_affinity_list
    done
    set_cpu_sleep 0
    for p in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
        echo schedutil >"$p" 2>/dev/null || echo ondemand >"$p" 2>/dev/null || true
    done
    rm -f "$DVFS_KO"
    [ -f /etc/systemd/system/rocknpu-tune.service ] && {
        systemctl disable --now rocknpu-tune.service >/dev/null 2>&1 || true
        rm -f /etc/systemd/system/rocknpu-tune.service /usr/local/sbin/rocknpu-tune
        systemctl daemon-reload
    }
    status
}

install_unit() {
    need_root
    install -m 0755 "$0" /usr/local/sbin/rocknpu-tune
    cat >/etc/systemd/system/rocknpu-tune.service <<'EOF'
[Unit]
Description=RockNPU system tuning (NPU IRQ routing, CPU idle, governors, NPU clock)
After=systemd-modules-load.service
Before=ollama.service

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/rocknpu-tune apply
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable --now rocknpu-tune.service
    status
}

case "$mode" in
    apply) apply ;;
    install) install_unit ;;
    dvfs) build_dvfs ;;
    restore) restore ;;
    status) status ;;
    *) echo "usage: $0 {apply|install|dvfs|restore|status}" >&2; exit 2 ;;
esac
