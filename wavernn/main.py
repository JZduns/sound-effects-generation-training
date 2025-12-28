import argparse
import logging
import os
from collections import defaultdict
from datetime import datetime
from time import time
from typing import List
import json

import torch
import torchaudio
from datasets import collate_factory, split_process_dataset
from losses import LongCrossEntropyLoss, MoLLoss
from processing import NormalizeDB
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torchaudio.models.wavernn import WaveRNN
from utils import count_parameters, MetricLogger, save_checkpoint
from audiomentations_wrapper import AudiomentationsWrapper
from audiomentations import Compose, PitchShift, Gain, TimeStretch


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint",
    )
    parser.add_argument(
        "--epochs",
        default=8000,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number")
    parser.add_argument(
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency in epochs",
    )
    parser.add_argument(
        "--dataset",
        default="ljspeech",
        choices=["ljspeech", "libritts", "basicaudiodataset"],
        type=str,
        help="select dataset to train with",
    )
    parser.add_argument("--batch-size", default=256, type=int, metavar="N", help="mini-batch size")
    parser.add_argument(
        "--learning-rate",
        default=1e-4,
        type=float,
        metavar="LR",
        help="learning rate",
    )
    parser.add_argument("--clip-grad", metavar="NORM", type=float, default=4.0)
    parser.add_argument(
        "--mulaw",
        default=True,
        action="store_true",
        help="if used, waveform is mulaw encoded",
    )
    parser.add_argument("--jit", default=False, action="store_true", help="if used, model is jitted")
    parser.add_argument(
        "--upsample-scales",
        default=[5, 5, 11],
        type=List[int],
        help="the list of upsample scales",
    )
    parser.add_argument(
        "--n-bits",
        default=8,
        type=int,
        help="the bits of output waveform",
    )
    parser.add_argument(
        "--sample-rate",
        default=22050,
        type=int,
        help="the rate of audio dimensions (samples per second)",
    )
    parser.add_argument(
        "--hop-length",
        default=275,
        type=int,
        help="the number of samples between the starts of consecutive frames",
    )
    parser.add_argument(
        "--win-length",
        default=1100,
        type=int,
        help="the length of the STFT window",
    )
    parser.add_argument(
        "--f-min",
        default=40.0,
        type=float,
        help="the minimum frequency",
    )
    parser.add_argument(
        "--min-level-db",
        default=-100,
        type=float,
        help="the minimum db value for spectrogam normalization",
    )
    parser.add_argument(
        "--n-res-block",
        default=10,
        type=int,
        help="the number of ResBlock in stack",
    )
    parser.add_argument(
        "--n-rnn",
        default=512,
        type=int,
        help="the dimension of RNN layer",
    )
    parser.add_argument(
        "--n-fc",
        default=512,
        type=int,
        help="the dimension of fully connected layer",
    )
    parser.add_argument(
        "--kernel-size",
        default=5,
        type=int,
        help="the number of kernel size in the first Conv1d layer",
    )
    parser.add_argument(
        "--n-freq",
        default=80,
        type=int,
        help="the number of spectrogram bins to use",
    )
    parser.add_argument(
        "--n-hidden-melresnet",
        default=128,
        type=int,
        help="the number of hidden dimensions of resblock in melresnet",
    )
    parser.add_argument(
        "--n-output-melresnet",
        default=128,
        type=int,
        help="the output dimension of melresnet",
    )
    parser.add_argument(
        "--n-fft",
        default=2048,
        type=int,
        help="the number of Fourier bins",
    )
    parser.add_argument(
        "--loss",
        default="crossentropy",
        choices=["crossentropy", "mol"],
        type=str,
        help="the type of loss",
    )
    parser.add_argument(
        "--seq-len-factor",
        default=5,
        type=int,
        help="the length of each waveform to process per batch = hop_length * seq_len_factor",
    )
    parser.add_argument(
        "--val-ratio",
        default=0.1,
        type=float,
        help="the ratio of waveforms for validation",
    )
    parser.add_argument(
        "--file-path",
        default="",
        type=str,
        help="the path of audio files",
    )
    parser.add_argument(
        "--normalization",
        default=True,
        action="store_true",
        help="if True, spectrogram is normalized",
    )
    parser.add_argument(
        "--weights",
        default="",
        type=str,
        metavar="PATH",
        help="path to state_dict",
    )

    
    return parser


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch):

    model.train()

    sums = defaultdict(lambda: 0.0)
    start1 = time()

    metric = MetricLogger("train_iteration")
    metric["epoch"] = epoch

    for waveform, specgram, target in data_loader:

        start2 = time()

        waveform = waveform.to(device)
        specgram = specgram.to(device)
        target = target.to(device)

        output = model(waveform, specgram)
        output, target = output.squeeze(1), target.squeeze(1)

        loss = criterion(output, target)
        loss_item = loss.item()
        sums["loss"] += loss_item
        metric["loss"] = loss_item

        optimizer.zero_grad()
        loss.backward()

        if args.clip_grad > 0:
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            sums["gradient"] += gradient.item()
            metric["gradient"] = gradient.item()

        optimizer.step()

        metric["iteration"] = sums["iteration"]
        metric["time"] = time() - start2
        metric()
        sums["iteration"] += 1

    avg_loss = sums["loss"] / len(data_loader)
    epoch_time = time() - start1

    metric = MetricLogger("train_epoch")
    metric["epoch"] = epoch
    metric["loss"] = sums["loss"] / len(data_loader)
    metric["gradient"] = avg_loss
    metric["time"] = epoch_time
    metric()
    logging.info(f"epoch: {epoch}, loss: {avg_loss}, time: {epoch_time}")
    


def validate(model, criterion, data_loader, device, epoch, scheduler):

    with torch.no_grad():

        model.eval()
        sums = defaultdict(lambda: 0.0)
        start = time()

        for waveform, specgram, target in data_loader:

            waveform = waveform.to(device)
            specgram = specgram.to(device)
            target = target.to(device)

            output = model(waveform, specgram)
            output, target = output.squeeze(1), target.squeeze(1)

            loss = criterion(output, target)
            sums["loss"] += loss.item()

        avg_loss = sums["loss"] / len(data_loader)
        scheduler.step(avg_loss)

        metric = MetricLogger("validation")
        metric["epoch"] = epoch
        metric["loss"] = avg_loss
        metric["Learning Rate"] = scheduler.get_last_lr()[0]
        metric["time"] = time() - start
        metric()

        return avg_loss


def main(args):

    devices = ["cuda" if torch.cuda.is_available() else "cpu"]

    logging.info("Start time: {}".format(str(datetime.now())))

    melkwargs = {
        "n_fft": args.n_fft,
        "power": 1,
        "hop_length": args.hop_length,
        "win_length": args.win_length,
    }

    transforms = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=args.sample_rate,
            n_mels=args.n_freq,
            f_min=args.f_min,
            mel_scale="slaney",
            norm="slaney",
            **melkwargs,
        ),
        NormalizeDB(min_level_db=args.min_level_db, normalization=args.normalization),
    )
    train_augmentations = Compose([PitchShift(min_semitones=-2, max_semitones=2), 
                                   Gain(min_gain_db=-3, max_gain_db=3), 
                                   TimeStretch(min_rate=0.9, max_rate=1.1)                                                               
                                   ])

    train_dataset, val_dataset = split_process_dataset(args, torch.nn.Sequential(AudiomentationsWrapper(train_augmentations, args.sample_rate), transforms), transforms)

    loader_training_params = {
        "num_workers": args.workers,
        "pin_memory": False,
        "shuffle": True,
        "drop_last": False,
    }
    loader_validation_params = loader_training_params.copy()
    loader_validation_params["shuffle"] = False

    collate_fn = collate_factory(args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        **loader_training_params,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        **loader_validation_params,
    )

    n_classes = 2**args.n_bits if args.loss == "crossentropy" else 30

    params = {
        "upsample_scales": args.upsample_scales,
        "n_classes": n_classes,
        "hop_length": args.hop_length,
        "n_res_block": args.n_res_block,
        "n_rnn": args.n_rnn,
        "n_fc": args.n_fc,
        "kernel_size": args.kernel_size,
        "n_freq": args.n_freq,
        "n_hidden": args.n_hidden_melresnet,
        "n_output": args.n_output_melresnet
    }
    params = json.dumps(params, indent=4)    
    with open("waveRNN_model_params.json", "w") as f:
        f.write(params)
    logging.info("model parameters saved as 'waveRNN_model_params.json'")

    model = WaveRNN(
        upsample_scales=args.upsample_scales,
        n_classes=n_classes,
        hop_length=args.hop_length,
        n_res_block=args.n_res_block,
        n_rnn=args.n_rnn,
        n_fc=args.n_fc,
        kernel_size=args.kernel_size,
        n_freq=args.n_freq,
        n_hidden=args.n_hidden_melresnet,
        n_output=args.n_output_melresnet,
    )

    if args.jit:
        model = torch.jit.script(model)

    

    n = count_parameters(model)
    logging.info(f"Number of parameters: {n}")

     # wczytanie wag
    if args.weights and os.path.isfile(args.weights):
        logging.info(f"Weights: loading '{args.weights}'")
        state_dict = torch.load(args.weights, weights_only=True)

        model.load_state_dict(state_dict)

    # Optimizer
    optimizer_params = {
        "lr": args.learning_rate,
    }

    optimizer = Adam(model.parameters(), **optimizer_params)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8)

    criterion = LongCrossEntropyLoss() if args.loss == "crossentropy" else MoLLoss()

    best_loss = 10.0

    # wczytanie checkpointu
    if args.checkpoint and os.path.isfile(args.checkpoint):
        logging.info(f"Checkpoint: loading '{args.checkpoint}'")
        checkpoint = torch.load(args.checkpoint, map_location=devices[0])

        args.start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]

        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(devices[0] )

        logging.info(f"Checkpoint: loaded '{args.checkpoint}' at epoch {checkpoint['epoch']}")
    else:
        logging.info("Checkpoint: not found")

        save_checkpoint(
            {
                "epoch": args.start_epoch,
                "state_dict": model.state_dict(),
                "best_loss": best_loss,
                "optimizer": optimizer.state_dict(),
            },
            False,
            args.checkpoint,
        )

    # usunięcie starych logów
    logs_file = "metrics_logs.txt"
    if os.path.isfile(logs_file):
        logging.info("Clearing old logs")
        with open(logs_file, "r+") as f:
            f.seek(0)
            f.truncate()

    model = torch.nn.DataParallel(model)
    model = model.to(devices[0], non_blocking=True)

    for epoch in range(args.start_epoch, args.epochs):

        train_one_epoch(
            model,
            criterion,
            optimizer,
            train_loader,
            devices[0],
            epoch,
        )        

        sum_loss = validate(model, criterion, val_loader, devices[0], epoch, scheduler)

        if not (epoch + 1) % args.print_freq or epoch == args.epochs - 1:

            is_best = sum_loss < best_loss
            best_loss = min(sum_loss, best_loss)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.module.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,                
                args.checkpoint
            )

    logging.info(f"End time: {datetime.now()}")


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    parser = arg_parser()
    args = parser.parse_args()
    main(args)
