# mice-fee

## Machine Learning the Entropy to Estimate Free Energy Differences without Sampling Transitions

### Yamin Ben-Shimon, Barak Hirshberg, Yohai Bar-Sinai

The tooling in this repository reproduces the data preparation and neural-network pipelines used for the entropy-learning workflow described in [arXiv:2510.24930](https://arxiv.org/abs/2510.24930). It includes scripts for packing LAMMPS trajectories into HDF5, voxelizing atomic configurations, and training mutual-information estimators on the generated tensors.

## HDF5 Packing Pipeline

This adds a streaming HDF5 writer that replaces the trajectory splitting step.

## Requirements

- **LAMMPS:** 15 Jun 2023 development build (`patch_15Jun2023-30-gc5d9f901d9-modified`) as used in the bundled simulation logs.
- **PLUMED:** 2.8.3 (`git f1e636b5b`) with the PairEntropy plugin enabled, matching the `plumed*.out` metadata under `data/simulation`.
- **Python packages:** the tooling and training scripts rely on `numpy`, `torch`, `pandas`, `h5py`, `matplotlib`, `seaborn`, and `scipy`. Install them via:
  ```bash
  pip install numpy torch pandas h5py matplotlib seaborn scipy
  ```

**Note:** All scripts should be run from the main repository directory (the directory containing `data/` and `train/`).

### One-liners

Build *solid + liquid* train/val with HDF5:
```bash
./data/bin/run_all.sh -D m_Na365 -t 365 -n 6
./data/bin/run_all.sh -D m_Al933 -t 933 -n 6 -e Al
```

Build a single phase + seeds, and write `coordinates.h5` inside the dataset:
```bash
./data/bin/run_phase.sh -d m_Na365_L_train -p liquid -t 365 -n 6 -s data/seeds/seeds_train -o coordinates.h5 -k 200
./data/bin/run_phase.sh -d m_Al933_S_train -p solid -t 933 -n 6 -s data/seeds/seeds_train -o coordinates.h5 -k 200 -e Al
```

- `-k` controls how many initial frames to skip (equilibration).
- The HDF5 file has a **group per seed** with a chunked, resizable dataset:
  `/<seed>/coordinates` shaped `(frames, 1024, 3)` in `float32`, default compression `lzf`.

### Notes

- Scripts automatically detect the repository root (directory containing `data/`) and resolve paths accordingly.
- Works with the original per-seed creators (no changes needed).
- Element-specific plumed files (e.g., `plumedNa.dat`, `plumedAl.dat`) are copied from `data/sim_templates/<element>/` into each dataset folder.
- Seeds files can be specified relative to `data/` directory or as absolute paths.
- For parallel seeds: `export XARGS_P="-P 2"` before running.

```bash
python data/tools/python/make_dataset.py \
  --h5 coordinates.h5 \
  --element Al \
  --bins 32 64 \
  --bf 1.0 \
  --outdir ./out \
  --dataset-key positions
```

```bash
# python data/tools/python/make_dataset.py --h5 data/simulation/m_Na365_L_train/coordinates.h5 --element Na --bins 32 --bf 0.3 --outdir coordinates
```

## Training The Neural Network

The training scripts now load `.npy` tensors from `data/coordinates` directly (no Weights & Biases required). By default the trainer looks for `coordinates_train_<ELEMENT>_bf<BIN>_bin<BINS>.npy` and the matching validation file, but you can override the filenames explicitly.

Example runs that train on the provided datasets:

```bash
# Sodium dataset
python train/mw_train.py \
  --train-file coordinates_train_Na_bf0.3_bin32 \
  --val-file   coordinates_val_Na_bf0.3_bin32 \
  --run-name   Na_bf0.3_bin32 \
  --bins 32 \
  --seed 42

# Aluminum dataset
python train/mw_train.py \
  --element Al \
  --bf 0.3 \
  --bins 32 \
  --run-name Al_bf0.3_bin32 \
  --seed 42
```
#python train/mw_train.py --train-file coordinates_train_Na_bf0.4_bin32 --val-file   coordinates_val_Na_bf0.4_bin32 --run-name   Na_bf0.4_bin32   --bins 32 --seed 42

Useful flags:

- `--data-dir` points to the directory that holds the `.npy` tensors (defaults to `data/coordinates`).
- `--element`/`--bf`/`--bins` can be used instead of `--train-file`/`--val-file` to auto-build filenames.
- `--nsamples` limits the number of examples per split if you want to dry-run the pipeline.
- Pass `--mice --dx 16 --dy 16 --dz 16` to enable the MICE-specific cropping heuristics.
- Artifacts land in `train/models/`, `train/results/`, and `train/logs/` (named after `--run-name` plus the chosen hyper-parameters) so you can keep checkpoints, metrics, and text summaries organized.

## Parsing Results

Parse training results from metrics files and create a DataFrame for analysis:

```bash
# Parse results from default folder (train/results/mice)
python train/parse_results.py

# Parse and save to CSV
python train/parse_results.py --output results_summary.csv

# Parse from custom folder
python train/parse_results.py --folder results/my_experiments
```

The script extracts hyperparameters from filenames (e.g., `Na_bf0.4_bin32_mice_dx16_dy16_dz16_s42_w16_..._metrics.npy`) and creates a DataFrame with training and validation MI values. You can also import it as a module:

```python
from train.parse_results import parse_results_folder
df = parse_results_folder("train/results/mice")
```
