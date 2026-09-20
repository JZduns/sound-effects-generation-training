"""
Skrypt dzieli dataset ESC-50 na foldery train i test z plikami przetworzonymi tak jak w datasecie ljspeech
oraz tworzy pliki z metadanymi: metadata.csv dla WaveRNN i filelist_train/val.txt dla HiFi-GAN
"""

import os, copy, csv, argparse, librosa
import pandas as pd
import soundfile as sf

parser = argparse.ArgumentParser()
parser.add_argument(
    "--annotations-file",
    default="ESC-50/meta/esc50.csv",
    help="path to annotations file",
)
parser.add_argument(
    "--audio-dir", default="ESC-50/audio", help="path to dir with audio files"
)
parser.add_argument(
    "--new-dataset-path", default="new_dataset", help="path for new dataset"
)
parser.add_argument(
    "--test-files-per-class", default=5, help="number of test files per class"
)
args = parser.parse_args()
train_data_dir = "train_data"
test_data_dir = "test_data"

annotations = pd.read_csv(args.annotations_file)
# filename, fold, target, category, esc10, src_file, take
dataset_targets = annotations["target"].unique()
dataset_classses = annotations["category"].unique()

metadata_train = {key: [] for key in dataset_targets}
metadata_test = copy.deepcopy(metadata_train)

test_classes_count = {key: 0 for key in dataset_targets}

os.makedirs(os.path.join(args.new_dataset_path, train_data_dir, "wavs"), exist_ok=True)
os.makedirs(os.path.join(args.new_dataset_path, test_data_dir, "wavs"), exist_ok=True)

for c in dataset_classses:
    os.makedirs(
        os.path.join(args.new_dataset_path, test_data_dir, "wavs", c), exist_ok=True
    )

for i, row in annotations.iterrows():
    file_name = row.iloc[0]
    target = row.iloc[2]
    label = row.iloc[3]

    # preprocessing
    y, sr = librosa.load(os.path.join(args.audio_dir, file_name), mono=True)
    new_sr = 22050
    y = librosa.resample(y=y, orig_sr=sr, target_sr=new_sr)
    y, _ = librosa.effects.trim(y=y)
    if librosa.get_duration(y=y, sr=new_sr) < 1:
        continue

    # podział na dane treningowe i testowe
    if test_classes_count[target] < args.test_files_per_class:
        path_to_save = os.path.join("wavs", label, file_name)
        metadata_test[target].append(
            [file_name.removesuffix(".wav"), label, path_to_save]
        )
        path_to_save = os.path.join(args.new_dataset_path, test_data_dir, path_to_save)
        test_classes_count[target] += 1
    else:
        path_to_save = os.path.join("wavs", file_name)
        metadata_train[target].append(
            [file_name.removesuffix(".wav"), label, path_to_save]
        )
        path_to_save = os.path.join(args.new_dataset_path, train_data_dir, path_to_save)

    sf.write(path_to_save, y, new_sr, subtype="PCM_16")

# zapis metadanych
path_to_save = os.path.join(args.new_dataset_path, train_data_dir, "metadata.csv")
print(f"saving train metadata in '{path_to_save}'")
with open(path_to_save, "w", newline="") as f:
    writer = csv.writer(f, delimiter="|")
    for c in metadata_train.keys():
        writer.writerows(metadata_train[c])

annotations_train = pd.read_csv(path_to_save, header=None, delimiter="|")
annotations_val_hifigan = annotations_train.groupby(1, group_keys=False).apply(
    lambda x: x.sample(frac=0.1, random_state=42)
)
annotations_train_hifigan = annotations_train.drop(annotations_val_hifigan.index)

path_to_save = os.path.join(args.new_dataset_path, train_data_dir, "filelist_train.txt")
print(f"saving train filelist in '{path_to_save}'")
annotations_train_hifigan[2].to_csv(path_to_save, index=False, header=False)
path_to_save = os.path.join(args.new_dataset_path, train_data_dir, "filelist_val.txt")
print(f"saving val filelist in '{path_to_save}'")
annotations_val_hifigan[2].to_csv(path_to_save, index=False, header=False)

path_to_save = os.path.join(args.new_dataset_path, test_data_dir, "metadata.csv")
print(f"saving test metadata in '{path_to_save}'")
with open(path_to_save, "w", newline="") as f:
    writer = csv.writer(f, delimiter="|")
    for c in metadata_test.keys():
        writer.writerows(metadata_test[c])
