"""
Skrypt tworzy mel-spektrogramy z danych testowych, a następnie zapisuje pliki z audio odtworzonym przy pomocy algorytmu Griffin-Lim.
Pliki zapisywane są wewnątrz folderu z datasetem w folderze griffin-lim/.
"""

import os, librosa, argparse
import pandas as pd
import soundfile as sf

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir-path", help="path to dataset dir")
args = parser.parse_args()

annotations = pd.read_csv(
    os.path.join(args.dataset_dir_path, "metadata.csv"), delimiter="|", header=None
)

for r in annotations.itertuples():
    file_path = os.path.join(args.dataset_dir_path, r[3])
    new_file_path = os.path.join(args.dataset_dir_path, "griffin-lim", r[2])
    os.makedirs(new_file_path, exist_ok=True)
    new_file_path = os.path.join(new_file_path, r[1] + ".wav")

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
    new_waveform = librosa.feature.inverse.mel_to_audio(
        S, sr=sample_rate, n_fft=1024, hop_length=256, win_length=1024
    )

    with open(new_file_path, mode="w") as f:
        sf.write(new_file_path, new_waveform, sample_rate, "PCM_16")
