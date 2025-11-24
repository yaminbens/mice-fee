#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "Usage: PHASE={solid|liquid} TEMP=<K> TIME_NS=<int> $0 <seed> <bindir>" >&2
  exit 2
fi

seed="$1"
bindir="$2"

: "${PHASE:?PHASE env var required (solid|liquid)}}"
: "${TEMP:?TEMP env var required (temperature in K)}}"
: "${TIME_NS:?TIME_NS env var required (ns)}}"

case "$PHASE" in
  solid)
    "$bindir/createSolidSim.sh" -t "$TEMP" -p solid -s "$seed" -n "$TIME_NS"
    ;;
  liquid)
    "$bindir/createLiquidSim.sh" -t "$TEMP" -p liquid -s "$seed" -n "$TIME_NS"
    ;;
  *)
    echo "Unknown PHASE='$PHASE' (expected 'solid' or 'liquid')" >&2
    exit 3
    ;;
esac
