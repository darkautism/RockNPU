#!/bin/sh
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "run as root: sudo $0 {apply|restore}" >&2
    exit 1
fi

mode="${1:-apply}"
case "$mode" in
    apply|restore) : ;;
    *) echo "usage: $0 {apply|restore}" >&2; exit 2 ;;
esac

irq_for() {
    dev="$1"
    irq=$(awk -v dev="$dev" '$0 ~ dev {gsub(":", "", $1); print $1; exit}' /proc/interrupts)
    [ -n "$irq" ] || { echo "missing IRQ for $dev" >&2; exit 1; }
    printf '%s\n' "$irq"
}

cpu_sleep_disable_for() {
    cpu="$1"
    for d in /sys/devices/system/cpu/cpu"$cpu"/cpuidle/state*; do
        [ -f "$d/name" ] || continue
        if [ "$(cat "$d/name")" = "cpu-sleep" ]; then
            printf '%s\n' "$d/disable"
            return 0
        fi
    done
    echo "no cpu-sleep idle state for CPU$cpu" >&2
    exit 1
}

irq0=$(irq_for fdab0000.npu)
irq1=$(irq_for fdac0000.npu)
irq2=$(irq_for fdad0000.npu)

case "$mode" in
    apply)
        echo 0 > "/proc/irq/$irq0/smp_affinity_list"
        echo 1 > "/proc/irq/$irq1/smp_affinity_list"
        echo 2 > "/proc/irq/$irq2/smp_affinity_list"
        for cpu in 0 1 2; do
            echo 1 > "$(cpu_sleep_disable_for "$cpu")"
        done
        ;;
    restore)
        nproc=$(getconf _NPROCESSORS_ONLN)
        last=$((nproc - 1))
        all="0-$last"
        for irq in "$irq0" "$irq1" "$irq2"; do
            echo "$all" > "/proc/irq/$irq/smp_affinity_list"
        done
        for cpu in 0 1 2; do
            echo 0 > "$(cpu_sleep_disable_for "$cpu")"
        done
        ;;
esac

echo "mode=$mode irqs=$irq0,$irq1,$irq2 affinity=$(cat /proc/irq/"$irq0"/smp_affinity_list),$(cat /proc/irq/"$irq1"/smp_affinity_list),$(cat /proc/irq/"$irq2"/smp_affinity_list)"
