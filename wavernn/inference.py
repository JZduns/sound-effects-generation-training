import argparse

import torch
import torchaudio
import json
import os
from torch.utils.data import DataLoader
from processing import NormalizeDB
from torchaudio.models.wavernn import WaveRNN
from torchaudio.transforms import MelSpectrogram
from wavernn_inference_wrapper import WaveRNNInferenceWrapper
from basic_audio_dataset import BasicAudioDataset

from time import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir-path",
        default="output",
        type=str,
        metavar="PATH",
        help="The path to output the reconstructed wav files.",
    )
    parser.add_argument(
        "--jit", default=False, action="store_true", help="If used, the model and inference function is jitted."
    )
    parser.add_argument("--no-batch-inference", default=False, action="store_true", help="Don't use batch inference.")
    parser.add_argument(
        "--no-mulaw", default=False, action="store_true", help="Don't use mulaw decoder to decoder the signal."
    )
    parser.add_argument(
        "--checkpoint-name",
        default="wavernn_10k_epochs_8bits_ljspeech",
        help="Select the WaveRNN checkpoint.",
    )
    parser.add_argument(
        "--batch-timesteps",
        default=100,
        type=int,
        help="The time steps for each batch. Only used when batch inference is used",
    )
    parser.add_argument(
        "--batch-overlap",
        default=5,
        type=int,
        help="The overlapping time steps between batches. Only used when batch inference is used",
    )
    parser.add_argument(
        "--model-params",
        default="waveRNN_model_params.json",
        help="File with model parameters",
    )
    parser.add_argument(
        "--dataset-path",
        default="dataset",
        help="path to dataset"
    )
    parser.add_argument(
        "--single",
        default=False,
        help="Use single file from dataset"
    )
    args = parser.parse_args()
    return args


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = BasicAudioDataset(args.dataset_path)
    _, sample_rate, _, _ = dataset[0]

    mel_kwargs = {
        "sample_rate": sample_rate,
        "n_fft": 2048,
        "f_min": 40.0,
        "n_mels": 80,
        "win_length": 1100,
        "hop_length": 275,
        "mel_scale": "slaney",
        "norm": "slaney",
        "power": 1,
    }
    transforms = torch.nn.Sequential(
        MelSpectrogram(**mel_kwargs),
        NormalizeDB(min_level_db=-100, normalization=True),
    )
    

    # model params
    with open(args.model_params, 'r') as f: 
        params = json.load(f)
    print("model parameters loaded")
    
    # state dict
    checkpoint = torch.load(args.checkpoint_name, weights_only=False)

    wavernn_model = WaveRNN(
        upsample_scales = params["upsample_scales"],
        n_classes = params["n_classes"],
        hop_length = params["hop_length"],
        n_res_block = params["n_res_block"],
        n_rnn = params["n_rnn"],
        n_fc = params["n_fc"],
        kernel_size = params["kernel_size"],
        n_freq = params["n_freq"],
        n_hidden = params["n_hidden"],
        n_output = params["n_output"]
    )

    wavernn_model.load_state_dict(checkpoint["state_dict"])
    wavernn_model = wavernn_model.eval().to(device)
    print("checkpoint loaded")
    
    wavernn_inference_model = WaveRNNInferenceWrapper(wavernn_model)

    if args.jit:
        wavernn_inference_model = torch.jit.script(wavernn_inference_model)

    data_loader = DataLoader(dataset, shuffle=True)

    for waveform, sample_rate, label, fileid in data_loader:
        
        mel_specgram = transforms(waveform.squeeze(0))

        if args.single: t0 = time()

        with torch.no_grad():
            output = wavernn_inference_model(
                mel_specgram.to(device),
                mulaw=(not args.no_mulaw),
                batched=(not args.no_batch_inference),
                timesteps=args.batch_timesteps,
                overlap=args.batch_overlap,
            )

        save_path = os.path.join(args.output_dir_path, label[0])
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, fileid[0] + ".wav")
        torchaudio.save(save_path, output, sample_rate=sample_rate.item())
        
        if args.single:
            print(f"generation took {time()-t0} s, file saved in {save_path}")            
            break


if __name__ == "__main__":
    args = parse_args()
    main(args)
