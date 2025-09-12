import csv
import os
from pathlib import Path
from typing import Tuple, Union

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset


class BasicAudioDataset(Dataset):

    def __init__(
        self,
        root: Union[str, Path],
        folder_in_archive: str = "wavs"
    ) -> None:

        self._parse_filesystem(root, folder_in_archive)

    def _parse_filesystem(self, root: str, folder_in_archive: str) -> None:
        root = Path(root)

        self._path = root / folder_in_archive
        self._metadata_path = root / "metadata.csv"

        if not os.path.exists(self._path):
            raise RuntimeError(
                f"The path {self._path} doesn't exist. "
                "Please check the ``root`` path"
            )

        with open(self._metadata_path, "r", newline="") as metadata:
            flist = csv.reader(metadata, delimiter="|", quoting=csv.QUOTE_NONE)
            self._flist = list(flist)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str]:
        line = self._flist[n]
        fileid, label, file_path = line
        fileid_audio = self._path / (fileid + ".wav")

        # Load audio
        waveform, sample_rate = torchaudio.load(fileid_audio)

        return (
            waveform,
            sample_rate,
            label
        )


    def __len__(self) -> int:
        return len(self._flist)
    