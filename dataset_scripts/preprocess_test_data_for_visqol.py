"""
Skrypt resampluje pliki do wymaganego SR, zapisuje w nowym folderze i tworzy plik batch_input.csv do użycia jako input dla VISQOL.
"""

import os, librosa, csv, argparse
import pandas as pd
import soundfile as sf

parser = argparse.ArgumentParser()
parser.add_argument("--original-dir", help="original audio dir path")
parser.add_argument("--generated-dir", help="generated audio dir path")
parser.add_argument("--new-dir", help="path for new dir")
parser.add_argument("--new-sr", default=16000, help="new sample rate")
args = parser.parse_args()

annotations = pd.read_csv(
    os.path.join(args.original_dir, "metadata.csv"), delimiter="|", header=None
)

for c in annotations[1].unique():
    os.makedirs(os.path.join(args.new_dir, "original", c), exist_ok=True)
    os.makedirs(os.path.join(args.new_dir, "generated", c), exist_ok=True)


def resample_save_file(path_to_file, path_to_save, target_sr):
    wav, sr = librosa.load(path_to_file, sr=None)
    wav = librosa.resample(y=wav, orig_sr=sr, target_sr=target_sr)
    sf.write(path_to_save, wav, args.new_sr, "PCM_16")


metadata = []

for i, row in annotations.iterrows():
    file_name = row.iloc[0]
    label = row.iloc[1]
    original_file_path = os.path.join(args.original_dir, row.iloc[2])
    generated_file_path = os.path.join(args.generated_dir, label, file_name + ".wav")
    assert os.path.isfile(original_file_path) and os.path.isfile(
        generated_file_path
    ), f"wrong path: {original_file_path} {generated_file_path}"

    original_new_file_path = os.path.join(
        args.new_dir, "original", label, file_name + ".wav"
    )
    resample_save_file(original_file_path, original_new_file_path, args.new_sr)
    generated_new_file_path = os.path.join(
        args.new_dir, "generated", label, file_name + ".wav"
    )
    resample_save_file(generated_file_path, generated_new_file_path, args.new_sr)

    metadata.append(
        {
            "reference": original_new_file_path.replace("\\", "/"),
            "degraded": generated_new_file_path.replace("\\", "/"),
        }
    )

    with open(os.path.join(args.new_dir, "batch_input.csv"), mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["reference", "degraded"])
        writer.writeheader()
        writer.writerows(metadata)
