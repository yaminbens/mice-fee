#!/usr/bin/env python3
"""
Parse MICE results from .npy metrics files and create a pandas DataFrame.

This script reads all metrics files from a results folder, parses the filenames
to extract hyperparameters, and creates a comprehensive DataFrame with all results.
"""

import os
import re
import argparse
import numpy as np
import pandas as pd


def parse_numeric(s):
    """Parse a string to float, handling scientific notation."""
    try:
        return float(s)
    except ValueError:
        return None


def parse_filename(filename):
    """
    Parse a metrics filename to extract hyperparameters.
    
    Expected format: Na_bf0.4_bin32_mice_dx16_dy16_dz16_s42_w16_b20_lr3e-05_ma3e-07_bs500_width20_m2.5_dfc0.3_dconv0.15_initxavier_s42_metrics.npy
    
    Args:
        filename: The filename to parse
        
    Returns:
        dict: Dictionary of parsed parameters
    """
    # Strip extension and split the filename by "_"
    stem = os.path.splitext(filename)[0]
    parts = stem.split("_")
    parsed = {}

    # Heuristic 1: leading token is often the element symbol (e.g. "Na", "Al").
    if parts:
        first = parts[0]
        # Accept short alphabetic tokens as potential element labels.
        if first.isalpha() and 1 <= len(first) <= 3:
            parsed["element"] = first

    # Heuristic 2: look for patterns like "m_Na365_S" anywhere in the name
    # to infer element and temperature from dataset-style prefixes.
    m = re.search(r"m_([A-Za-z]+)(\d+)_", stem)
    if m:
        elem, temp = m.group(1), m.group(2)
        parsed.setdefault("element", elem)
        try:
            parsed["temperature"] = int(temp)
        except ValueError:
            # If it doesn't parse cleanly, just leave temperature unset.
            pass
    
    # Iterate over each part to find key-value pairs
    # Order matters: check more specific patterns first
    for i, part in enumerate(parts):
        if part.startswith("bf"):
            parsed["bf"] = parse_numeric(part[2:])
        elif part.startswith("dconv"):
            parsed["d_conv"] = parse_numeric(part[5:])
        elif part.startswith("dfc"):
            parsed["d_fc"] = parse_numeric(part[3:])
        elif part.startswith("bin") and len(part) > 3 and part[3:].isdigit():
            # Handle bin parameter (single number like bin32)
            # Make sure it's not biny or binz
            if not part.startswith("biny") and not part.startswith("binz"):
                parsed["bin"] = int(part[3:])
        elif part.startswith("dx"):
            parsed["binx"] = int(part[2:])
        elif part.startswith("dy"):
            parsed["biny"] = int(part[2:])
        elif part.startswith("dz"):
            parsed["binz"] = int(part[2:])
        elif part in {"S", "L"}:
            # Optional single-letter phase label, if present in run name
            parsed["phase"] = part
        elif part.startswith("width"):
            parsed["width"] = int(part[5:])
        elif part.startswith("bs"):
            parsed["bs"] = int(part[2:])
        elif part.startswith("w") and len(part) > 1 and part[1:].isdigit():
            parsed["w"] = int(part[1:])
        elif part.startswith("ma"):
            # Handle scientific notation like ma3e-07
            ma_str = part[2:]
            parsed["ma"] = parse_numeric(ma_str)
        elif part.startswith("lr"):
            # Handle scientific notation like lr3e-05
            lr_str = part[2:]
            parsed["lr"] = parse_numeric(lr_str)
        elif part.startswith("m") and len(part) > 1:
            # Check if it's a numeric value (handles m2.5)
            m_str = part[1:]
            if m_str.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit() or ('e' in m_str.lower()):
                parsed["m"] = parse_numeric(m_str)
        elif part.startswith("s") and len(part) > 1 and part[1:].isdigit():
            parsed["seed"] = int(part[1:])
        elif part.startswith("init"):
            parsed["init"] = part[4:]  # Extract everything after "init"
        elif part == "mice":
            parsed["method"] = "mice"
    
    # Calculate total bin dimensions if dx, dy, dz are present
    if "binx" in parsed and "biny" in parsed and "binz" in parsed:
        parsed["dims"] = parsed["binx"] * parsed["biny"] * parsed["binz"]
    
    return parsed


def load_metrics(file_path, k=10000):
    """
    Load metrics from a .npy file and compute averages.
    
    Args:
        file_path: Path to the .npy file
        k: Number of last values to average (default: 10000)
        
    Returns:
        tuple: (train_MI, val_MI) - average mutual information values
    """
    try:
        data = np.load(file_path)
        
        # Take the average of the last k values
        if len(data) >= 2 and len(data[0]) >= k:
            ts = np.mean(data[0][-k:])
            vs = np.mean(data[1][-k:])
        elif len(data) >= 2:
            # If less than k values, use all available
            ts = np.mean(data[0])
            vs = np.mean(data[1])
        else:
            ts = np.mean(data[0]) if len(data) > 0 else np.nan
            vs = np.nan
        
        return ts, vs
    except Exception as e:
        raise Exception(f"Error loading {file_path}: {e}")


def parse_results_folder(folder_path, k=10000, verbose=True):
    """
    Parse all metrics files in a folder and create a DataFrame.
    
    Args:
        folder_path: Path to the folder containing .npy metrics files
        k: Number of last values to average for MI calculation (default: 10000)
        verbose: Whether to print progress and errors (default: True)
        
    Returns:
        pd.DataFrame: DataFrame containing all parsed results
    """
    results = []
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Iterate over the files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith("_metrics.npy"):
            parsed = parse_filename(filename)
            
            if len(parsed) > 5:
                # Load the metrics file
                file_path = os.path.join(folder_path, filename)
                try:
                    ts, vs = load_metrics(file_path, k=k)
                    
                    # Create result dictionaries (column order chosen to match pd_MICE.csv)
                    result_dict = {
                        "bf": parsed.get("bf"),
                        "bin": parsed.get("bin"),
                        "binx": parsed.get("binx"),
                        "biny": parsed.get("biny"),
                        "binz": parsed.get("binz"),
                        "dims": parsed.get("dims"),
                        "w": parsed.get("w"),
                        "bs": parsed.get("bs"),
                        "lr": parsed.get("lr"),
                        "ma": parsed.get("ma"),
                        "m": parsed.get("m"),
                        "d_fc": parsed.get("d_fc"),
                        "d_conv": parsed.get("d_conv"),
                        "width": parsed.get("width"),
                        "seed": parsed.get("seed"),
                        "init": parsed.get("init"),
                        "method": parsed.get("method", "mice"),
                        "MI": ts,
                        "style": "train",
                        "phase": parsed.get("phase"),
                        "element": parsed.get("element"),
                        "temperature": parsed.get("temperature"),
                    }
                    results.append(result_dict)
                    
                    result_dict_val = result_dict.copy()
                    result_dict_val["MI"] = vs
                    result_dict_val["style"] = "val"
                    results.append(result_dict_val)
                    
                except Exception as e:
                    if verbose:
                        print(f"Error loading {filename}: {e}")
                    continue
    
    # Convert results to a Pandas DataFrame
    df = pd.DataFrame(results)
    
    if verbose:
        print(f"Successfully parsed {len(results)} result entries from {len(results)//2} files")
    
    return df


def main():
    """Main function to run the script from command line."""
    parser = argparse.ArgumentParser(
        description="Parse MICE results from .npy metrics files and create a pandas DataFrame"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="results/mice",
        help="Path to the folder containing metrics files (default: results/mice)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the DataFrame as CSV (optional)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10000,
        help="Number of last values to average for MI calculation (default: 10000)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output messages"
    )
    
    args = parser.parse_args()
    
    # Parse results
    df = parse_results_folder(args.folder, k=args.k, verbose=not args.quiet)
    
    if not args.quiet:
        print(f"\nTotal rows: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nSummary Statistics:")
        print(df.describe())
    
    # Save to CSV if requested
    if args.output:
        df.to_csv(args.output, index=False)
        if not args.quiet:
            print(f"\nDataFrame saved to {args.output}")
    
    return df


if __name__ == "__main__":
    df = main()

