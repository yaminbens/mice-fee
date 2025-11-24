#!/usr/bin/env python3
import argparse, os, re, sys, glob, h5py, numpy as np

HDR_TS = "ITEM: TIMESTEP"
HDR_N  = "ITEM: NUMBER OF ATOMS"
HDR_BB = "ITEM: BOX BOUNDS"
HDR_AT = "ITEM: ATOMS"

def parse_dump_frames(path, skip_frames=0):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        line = f.readline()
        frame = -1
        n_atoms = None
        while line:
            if not line.startswith("ITEM:"):
                line = f.readline(); continue

            if line.startswith(HDR_TS):
                _ = f.readline()  # timestep value
                frame += 1
                line = f.readline(); continue

            if line.startswith(HDR_N):
                n_atoms = int(f.readline().strip())
                line = f.readline(); continue

            if line.startswith(HDR_BB):
                _ = f.readline(); _ = f.readline(); _ = f.readline()
                line = f.readline(); continue

            if line.startswith(HDR_AT):
                cols = line.strip().split()
                try:
                    id_idx = cols.index("id") - 2
                    x_idx = cols.index("x") - 2
                    y_idx = cols.index("y") - 2
                    z_idx = cols.index("z") - 2
                except ValueError:
                    raise RuntimeError(f"Expected id/x/y/z columns in: {line.strip()}")
                if n_atoms is None:
                    raise RuntimeError("NUMBER OF ATOMS header missing before ATOMS block")

                import numpy as np
                ids = np.empty((n_atoms,), dtype=np.int64)
                pos = np.empty((n_atoms, 3), dtype=np.float32)
                for i in range(n_atoms):
                    parts = f.readline().split()
                    if not parts:
                        raise RuntimeError(f"Unexpected EOF while reading atoms for frame {frame}")
                    ids[i] = int(parts[id_idx])
                    pos[i, 0] = float(parts[x_idx])
                    pos[i, 1] = float(parts[y_idx])
                    pos[i, 2] = float(parts[z_idx])

                if frame >= skip_frames:
                    order = np.argsort(ids, kind="stable")
                    yield (frame - skip_frames), pos[order]
                line = f.readline(); continue

            line = f.readline()

def pack_dataset_to_h5(dataset_dir, seeds, output, phase=None, temperature=None, compression="lzf", chunk=128, skip_frames=200):
    import h5py
    with h5py.File(output, "w") as h5:
        h5.attrs["format"] = "positions"
        if phase is not None:
            h5.attrs["phase"] = str(phase)
        if temperature is not None:
            h5.attrs["temperature_K"] = float(temperature)

        for seed in seeds:
            seed = seed.strip()
            if not seed:
                continue
            seed_dir = os.path.join(dataset_dir, seed)
            import glob
            dump_glob = glob.glob(os.path.join(seed_dir, "dump", "dump*.lammpstrj"))
            if not dump_glob:
                print(f"[WARN] No dump found for seed {seed}", file=sys.stderr)
                continue
            dump_path = dump_glob[0]

            grp = h5.create_group(seed)
            n_atoms = None
            dset = None
            frames_written = 0
            for fidx, pos in parse_dump_frames(dump_path, skip_frames=skip_frames):
                if n_atoms is None:
                    n_atoms = pos.shape[0]
                    dset = grp.create_dataset(
                        "positions",
                        shape=(0, n_atoms, 3),
                        maxshape=(None, n_atoms, 3),
                        chunks=(chunk, n_atoms, 3),
                        dtype="float32",
                        compression=compression
                    )
                    grp.attrs["n_atoms"] = n_atoms
                    if temperature is not None:
                        grp.attrs["temperature_K"] = float(temperature)
                    if phase is not None:
                        grp.attrs["phase"] = str(phase)

                dset.resize((frames_written + 1, n_atoms, 3))
                dset[frames_written, :, :] = pos
                frames_written += 1

            grp.attrs["frames"] = frames_written
            if frames_written == 0:
                print(f"[WARN] No frames written for seed {seed} (after skipping)", file=sys.stderr)

def read_seed_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def main():
    ap = argparse.ArgumentParser(description="Pack LAMMPS lammpstrj positions into a single HDF5 (group per seed).")
    ap.add_argument("--dataset", "-d", required=True, help="Dataset directory containing <seed>/dump/dump*.lammpstrj")
    ap.add_argument("--seeds-file", "-s", required=True, help="Path to seed list file used for the dataset")
    ap.add_argument("--output", "-o", default="positions.h5", help="Output HDF5 filename (created inside --dataset)")
    ap.add_argument("--phase", "-p", choices=["solid","liquid"], help="Phase metadata to attach")
    ap.add_argument("--temp", "-t", type=float, help="Temperature in K to attach as metadata")
    ap.add_argument("--skip-frames", "-k", type=int, default=200, help="Frames to skip at start (equilibration)")
    ap.add_argument("--chunk", type=int, default=128, help="Chunk length in frames")
    ap.add_argument("--no-compress", action="store_true", help="Disable compression")
    args = ap.parse_args()

    seeds = read_seed_list(args.seeds_file)
    output_path = os.path.join(args.dataset, args.output)
    compression = None if args.no_compress else "lzf"
    pack_dataset_to_h5(args.dataset, seeds, output_path, phase=args.phase, temperature=args.temp, compression=compression, chunk=args.chunk, skip_frames=args.skip_frames)
    print(f"✔ Wrote {output_path}")

if __name__ == "__main__":
    main()
