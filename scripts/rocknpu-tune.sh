#!/bin/sh
# System settings that make RockNPU faster. Nothing here changes the kernel
# or any file on disk except the optional systemd unit; `restore` undoes it.
#
#   sudo ./scripts/rocknpu-tune.sh apply     # apply now (until reboot)
#   sudo ./scripts/rocknpu-tune.sh install   # apply now and at every boot (systemd)
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
set -eu

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

apply() {
    need_root
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
Description=RockNPU system tuning (NPU IRQ routing, CPU idle, governors)
After=multi-user.target

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
    restore) restore ;;
    status) status ;;
    *) echo "usage: $0 {apply|install|restore|status}" >&2; exit 2 ;;
esac
