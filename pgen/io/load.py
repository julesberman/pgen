import glob
import os
import os.path
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf


def load_single(problem, name, out_dir="./presults"):

    out_dir = Path(out_dir)
    if name == "recent":
        list_of_files = glob.glob(f"{out_dir}/{problem}/single/*")
        file_path = max(list_of_files, key=os.path.getctime)
        file_path = Path(file_path)
        print(f"loading {file_path}")
    else:
        file_path = out_dir / problem / "single" / name

    cfg = OmegaConf.load(file_path / ".hydra/config.yaml")

    res_path = file_path / "result.pkl"
    df = None
    if os.path.isfile(res_path):
        df = pd.read_pickle(res_path)
    else:
        print("no file ", file_path)

    return cfg, df


def load_multi(problem, name, out_dir="./presults"):

    out_dir = Path(out_dir)
    parent_dir = out_dir / problem / "multi" / name

    rows, cfgs = [], []

    for item in os.listdir(parent_dir):
        item_path = parent_dir / item
        if os.path.isdir(item_path) and (
            item.isdigit() or item.split("_")[-1].isdigit()
        ):
            try:
                row = pd.read_pickle(item_path / "result.pkl")

                rows.append(row)
                cfg = OmegaConf.load(item_path / ".hydra/config.yaml")
                cfgs.append(cfg)
            except:
                print("did not load: ", item_path)

    df = pd.DataFrame(rows)
    return cfgs, df


def load_multi_recur(problem, name, out_dir="./presults"):

    out_dir = Path(out_dir)
    parent_dir = out_dir / problem / "multi" / name

    rows = []
    # Walk through directory recursively
    for root, _, files in os.walk(parent_dir):
        for file in files:
            # Check if filename matches 'result*.pkl' pattern
            if file.startswith("result") and file.endswith(".pkl"):
                file_path = os.path.join(root, file)
                try:
                    # Load pickle file and append to list
                    file_path = Path(file_path)
                    row = pd.read_pickle(file_path)
                    rows.append(row)
                    # cfg = OmegaConf.load(file_path / ".hydra/config.yaml")
                    # cfgs.append(cfg)
                except Exception as e:
                    print(f"Could not load {file_path}: {e}")

    df = pd.DataFrame(rows)
    print(f"{len(df)} rows")
    return df


def get_min_row(df, col):
    return df[df[col] == df[col].min()].iloc[0]
