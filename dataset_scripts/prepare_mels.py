"""
Skrypt zapisuje dane treningowe/testowe w formie mel-spektrogramów,
oraz zapisuje metadane dla HiFi-GAN w pliku mels_metadata.tsv.
Pliki .npy (treningowe) / .pt (testowe) zapisywane są wewnątrz folderu z datasetem w folderze mels/.
"""

import os, librosa, argparse, torch
import numpy as np
import pandas as pd


def save_melspec(file_path, new_file_path, save_as_torch=False):
    waveform, sample_rate = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        fmin=0,
        fmax=8000,
    )
    S = np.log(np.clip(S, a_min=1e-5, a_max=None))
    if save_as_torch:
        torch.save(torch.from_numpy(S), new_file_path)
    else:
        np.save(new_file_path, S)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir-path", help="path to dataset dir")
parser.add_argument(
    "--data-group",
    choices=["train", "test"],
    default="test",
    help="is dataset for training or testing",
)
args = parser.parse_args()

annotations = pd.read_csv(
    os.path.join(args.dataset_dir_path, "metadata.csv"), delimiter="|", header=None
)

os.makedirs(os.path.join(args.dataset_dir_path, "mels"), exist_ok=True)
new_annotations = pd.DataFrame({"mel": [], "output": [], "label": [], "text": []})
for r in annotations.itertuples():
    file_path = os.path.join(args.dataset_dir_path, r[3])

    if args.data_group == "test":
        new_file_path = os.path.join(args.dataset_dir_path, "mels", r[2])
        os.makedirs(new_file_path, exist_ok=True)
        new_file_path = os.path.join(new_file_path, r[1] + ".pt")
    else:
        new_file_path = os.path.join(args.dataset_dir_path, "mels", r[1] + ".npy")

    waveform, sample_rate = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        fmin=0,
        fmax=8000,
    )
    S = np.log(np.clip(S, a_min=1e-5, a_max=None))

    if args.data_group == "test":
        save_melspec(file_path, new_file_path, save_as_torch=True)
    else:
        save_melspec(file_path, new_file_path, save_as_torch=False)

    if args.data_group == "test":
        mel = os.path.join("mels", r[2], r[1] + ".pt").replace("\\", "/")
    else:
        mel = os.path.join("mels", r[1] + ".npy").replace("\\", "/")

    new_r = {"mel": mel, "output": r[1] + ".wav", "label": r[2], "text": "''"}
    new_annotations.loc[len(new_annotations)] = new_r

new_annotations.to_csv(
    os.path.join(args.dataset_dir_path, "mels_metadata.tsv"), sep="\t", index=False
)
