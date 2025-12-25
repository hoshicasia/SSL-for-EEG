import glob
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta

import h5py
import mne
import numpy as np
from tqdm import tqdm


def install_awscli():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "awscli"])


def download_siena_data(target_dir="siena"):
    os.makedirs(target_dir, exist_ok=True)
    original_dir = os.getcwd()
    os.chdir(target_dir)
    subprocess.check_call(
        [
            "aws",
            "s3",
            "sync",
            "--no-sign-request",
            "s3://physionet-open/siena-scalp-eeg/1.0.0/",
            "./",
        ]
    )
    os.chdir(original_dir)


TIME_RE = re.compile(r"(\d{1,2})[.:](\d{2})\.(\d{2})")


def extract_time(line):
    m = TIME_RE.search(line)
    h, mnt, s = m.groups()
    h = h.zfill(2)

    return datetime.strptime(f"{h}.{mnt}.{s}", "%H.%M.%S")


def offset_sec(t0, t1):
    if t1 < t0:
        t1 += timedelta(days=1)
    delta = (t1 - t0).total_seconds()
    if delta < 0:
        delta += 24 * 3600  # добавлю сутки, если время перескочило на новый день
    return delta


def parse_seizure_list(txt_path):
    records = []
    cur = {}
    reg_start = None

    with open(txt_path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if line.startswith("File name:"):
                cur["file"] = line.split(":")[1].strip()

            elif line.startswith("Registration start time:"):
                reg_start = extract_time(line)

            elif line.startswith("Seizure start time:"):
                cur["sz_start"] = extract_time(line)

            elif line.startswith("Seizure end time:"):
                cur["sz_end"] = extract_time(line)

                if reg_start is None or "sz_start" not in cur:
                    cur = {}
                    continue

                records.append(
                    {
                        "file": cur["file"],
                        "start": offset_sec(reg_start, cur["sz_start"]),
                        "end": offset_sec(reg_start, cur["sz_end"]),
                    }
                )

                cur = {}

    return records


def load_edf_windows(edf_path, seizures, win_sec=10, step_sec=5, sfreq=256):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick_types(eeg=True)
    if raw.info["sfreq"] != sfreq:
        raw.resample(sfreq)

    X = raw.get_data()
    sf = raw.info["sfreq"]

    win = int(win_sec * sf)
    step = int(step_sec * sf)

    file_seizures = [
        (s["start"], s["end"])
        for s in seizures
        if os.path.basename(s["file"]) == os.path.basename(edf_path)
    ]

    windows, labels, starts, ends = [], [], [], []

    for i in range(0, X.shape[1] - win, step):
        t0 = i / sf
        t1 = (i + win) / sf

        y = 0
        for s0, s1 in file_seizures:
            if not (t1 < s0 or t0 > s1):
                y = 1
                break

        windows.append(X[:, i : i + win].astype(np.float32))
        labels.append(y)
        starts.append(i)
        ends.append(i + win)

    return (
        np.stack(windows),
        np.array(labels, dtype=np.uint8),
        starts,
        ends,
        raw.ch_names,
    )


if __name__ == "__main__":
    install_awscli()
    download_siena_data(target_dir="siena")
    seizures = []
    for txt in glob.glob("/**/Seizures-list-*.txt", recursive=True):
        seizures.extend(parse_seizure_list(txt))

    out_path = "siena_femba.h5"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as h5:
        first = True
        n_channels = 22
        base_dir = "/siena"

        for subj in tqdm(sorted(os.listdir(base_dir))):
            subj_path = os.path.join(base_dir, subj)
            if not os.path.isdir(subj_path):
                continue
            for edf in glob.glob(f"{subj_path}/*.edf"):
                X, y, s, e, ch = load_edf_windows(edf, seizures)

                if X.shape[0] == 0:
                    continue

                if first:
                    h5.create_dataset(
                        "data",
                        shape=(0, n_channels, X.shape[2]),
                        maxshape=(None, n_channels, X.shape[2]),
                        chunks=(1, n_channels, X.shape[2]),
                        dtype="float32",
                    )
                    h5.create_dataset(
                        "labels", shape=(0,), maxshape=(None,), dtype="uint8"
                    )
                    h5.create_dataset(
                        "subject",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=h5py.special_dtype(vlen=str),
                    )
                    h5.create_dataset(
                        "file",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=h5py.special_dtype(vlen=str),
                    )
                    h5.create_dataset(
                        "win_start", shape=(0,), maxshape=(None,), dtype="int64"
                    )
                    h5.create_dataset(
                        "win_end", shape=(0,), maxshape=(None,), dtype="int64"
                    )
                    h5.create_dataset(
                        "channels",
                        data=np.array(ch, dtype=object),
                        dtype=h5py.special_dtype(vlen=str),
                    )
                    first = False

                n = X.shape[0]
                idx = slice(h5["data"].shape[0], h5["data"].shape[0] + n)

                for k in ["data", "labels", "subject", "file", "win_start", "win_end"]:
                    h5[k].resize(h5[k].shape[0] + n, axis=0)

                if X.shape[1] >= n_channels:
                    X_trimmed = X[:, :n_channels, :]
                else:
                    X_trimmed = np.zeros(
                        (X.shape[0], n_channels, X.shape[2]), dtype=X.dtype
                    )
                    X_trimmed[:, -X.shape[1] :, :] = X

                h5["data"][idx] = X_trimmed
                h5["labels"][idx] = y
                h5["subject"][idx] = subj
                h5["file"][idx] = os.path.basename(edf)
                h5["win_start"][idx] = s
                h5["win_end"][idx] = e
