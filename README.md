# Generacja efektów dźwiękowych z wykorzystaniem sieci neuronowych
Repozytorium zawiera skrypty i notatniki do treningu i inferencji modeli sieci neuronowych oraz obliczania metryk jakości.

## Przygotowanie danych
W folderze _dataset_scripts/_ znajdują się skrypty do przygotowania danych ze zbioru esc-50:
- _preprocess_dataset_ - zmienia SR na 22050 Hz, wycina fragmenty ciszy oraz dzieli zbiór na dane treningowe i testowe
- _prepare_griffin-lim_ - tworzy mel-spektrogramy z danych testowych, a następnie zapisuje audio odtworzone przy pomocy algorytmu Griffin-Lim
- _prepare_mels_ - zapisuje dane treningowe/testowe w formie mel-spektrogramów
- _preprocess_test_data_for_visqol_ - zmienia SR, zapisuje pliki w nowym folderze i tworzy plik _batch_input.csv_ dla VISQOL

## Wavernn
Folder _wavernn/_ zawiera zestaw skryptów do treningu i inferencji modelu WaveRNN oparty na przykładzie zawartym w [repozytorium TorchAudio](https://github.com/pytorch/audio/tree/main/examples/pipeline_wavernn). 

Przykładowe uruchomienie treningu:
```
python main.py \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --n-freq 80 \
    --loss "crossentropy" \
    --n-bits 8 \
    --workers 0 \
    --epochs 50 \
    --print-freq 5 \
    --file-path "./dataset/train_data" \
    --checkpoint "checkpoints/wavernn_checkpoint" \
    --dataset "basicaudiodataset" \
    --scheduler "CosineAnnealingLR" \
    --augmentations
```
W trakcie treningu zapisywane są bieżący i najlepszy checkpoint (pliki .pth), parametry modelu (plik .json) oraz logi (plik .txt).

Przykładowe uruchomienie inferencji:
```
python inference.py \
    --dataset-path ./dataset/test_data \
    --checkpoint-name ./checkpoints/model_best.pth.tar \
    --model-params ./checkpoints/waveRNN_model_params.json \
    --output-dir-path ./output
```

## HiFi-GAN
Folder _hifi-gan/_ zawiera notatniki .ipynb do treningu i inferencji modelu HiFi-GAN w implementacji z [repozytorium NVIDIA Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/HiFiGAN)

Przykładowe uruchomienie treningu:
```
python train.py \
    --cuda \
    --dataset_path ./data/dataset/train_data/ \
    --training_files ./data/filelists/metadata_train.txt \
    --validation_files ./data/filelists/metadata_val.txt \
    --output ./results \
    --checkpoint_interval 5 \
    --epochs 50 \
    --batch_size 16 \
    --learning_rate 0.0003 \
    --lr_decay 0.9998 \
    --validation_interval 1 \
    --step_logs_interval 20 \
    --resblock_kernel_sizes [3,7,11]
```
W trakcie treningu zapisywane są checkpointy generatora i dyskryminatora (pliki .pt) oraz logi (plik .json).
W celu wykonania fine-tuningu należy dodatkowo dodać folder z danymi w formie mel-spektrogramów oraz flagi `--fine_tuning` i `--input_mels_dir ./data/train_data/mels`.

Przykładowe uruchomienie inferencji:
```
python inference.py \
    --output audio/test \
    --hifigan weights/hifigan_gen_checkpoint_50.pt \
    -sr 22050 \
    --dataset-path data/test_data/ \
    --affinity disabled \
    --input data/test_data/mels_metadata.tsv
```

## Ocena jakości
Funkcje do rysowania mel-spektrogramów oraz obliczania metryk jakości i deskryptorów dźwięku zostały zebrane w notatniku _evaluation.ipynb_.