# HDF5 Packing Pipeline

This adds a streaming HDF5 writer that replaces the trajectory splitting step.

## One-liners

Build *solid + liquid* train/val with HDF5:
```bash
./bin/run_all_h5.sh -D m_Na365 -t 365 -n 6
```

Build a single phase + seeds, and write `positions.h5` inside the dataset:
```bash
./bin/run_phase_h5.sh -d m_Na365_S -p solid -t 365 -n 6 -s seeds/seeds20 -o positions.h5 -k 200
```

- `-k` controls how many initial frames to skip (equilibration).
- The HDF5 file has a **group per seed** with a chunked, resizable dataset:
  `/<seed>/positions` shaped `(frames, 1024, 3)` in `float32`, default compression `lzf`.

## Notes
- Works with the original per-seed creators (no changes needed).
- Keep `plumed.dat` at repo root; it's copied into each dataset folder.
- For parallel seeds: `export XARGS_P="-P 2"` before running.
