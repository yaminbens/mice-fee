#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $0 -d <dataset_name> -p <solid|liquid> -t <temp_K> -n <time_ns> -s <seeds_file> [-o <output.h5>] [-k <skip_frames>] [-b <bindir>]
Example:
  $0 -d m_Na365_S -p solid -t 365 -n 6 -s seeds/seeds20 -o positions.h5 -k 200
USAGE
}

dataset=""
phase=""
temp=""
time_ns=""
seeds_file=""
output="positions.h5"
skip_frames=200
bindir="$(dirname "$0")"

while getopts "d:p:t:n:s:o:k:b:h" opt; do
  case $opt in
    d) dataset="$OPTARG";;
    p) phase="$OPTARG";;
    t) temp="$OPTARG";;
    n) time_ns="$OPTARG";;
    s) seeds_file="$OPTARG";;
    o) output="$OPTARG";;
    k) skip_frames="$OPTARG";;
    b) bindir="$OPTARG";;
    h) usage; exit 0;;
    *) usage; exit 1;;
  esac
done

if [ -z "${dataset}" ] || [ -z "${phase}" ] || [ -z "${temp}" ] || [ -z "${time_ns}" ] || [ -z "${seeds_file}" ]; then
  usage; exit 1
fi

mkdir -p "${dataset}"

# Copy plumed.dat if available next to script or at repo root
if [ -f "plumed.dat" ] && [ ! -f "${dataset}/plumed.dat" ]; then
  cp "plumed.dat" "${dataset}/plumed.dat"
elif [ -f "$(dirname "$0")/../plumed.dat" ] && [ ! -f "${dataset}/plumed.dat" ]; then
  cp "$(dirname "$0")/../plumed.dat" "${dataset}/plumed.dat"
fi

pushd "${dataset}" >/dev/null

export PHASE="${phase}"
export TEMP="${temp}"
export TIME_NS="${time_ns}"

XARGS_P="${XARGS_P:-}"

if [ ! -f "../${seeds_file}" ] && [ -f "${seeds_file}" ]; then
  true
elif [ ! -f "${seeds_file}" ] && [ -f "../${seeds_file}" ]; then
  seeds_file="../${seeds_file}"
fi

if [ ! -f "${seeds_file}" ]; then
  echo "Seeds file not found: ${seeds_file}" >&2
  exit 5
fi

# Run per-seed jobs
cat "${seeds_file}" | xargs ${XARGS_P} -I {} bash -c '"$0/seedjob.sh" "$1" "$0"' "$bindir" {}

popd >/dev/null

# Pack to HDF5 (group per seed)
python3 "$bindir/pack_positions_h5.py" -d "${dataset}" -s "${seeds_file}" -o "${output}" -p "${phase}" -t "${temp}" -k "${skip_frames}"

echo "✔ Phase '${phase}' dataset built and packed: ${dataset}/${output}"
