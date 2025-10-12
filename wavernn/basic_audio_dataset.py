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
        root: Union[str, Path]
    ) -> None:

        self._parse_filesystem(root)

    def _parse_filesystem(self, root: str) -> None:
        root = Path(root)

        self._path = root
        self._metadata_path = os.path.join(root, "metadata.csv")

        if not os.path.exists(self._path):
            raise RuntimeError(
                f"The path {self._path} doesn't exist. "
                "Please check the ``root`` path"
            )
        
        if not os.path.isfile(self._metadata_path):
            raise RuntimeError(
                f"Metadata file not found in {self._metadata_path}"
            )

        with open(self._metadata_path, "r", newline="") as metadata:
            flist = csv.reader(metadata, delimiter="|", quoting=csv.QUOTE_NONE)
            self._flist = list(flist)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str]:
        line = self._flist[n]
        fileid, label, file_path = line
        file_path = file_path.replace("\\", "/")
        fileid_audio = os.path.join(self._path, file_path)

        # Load audio
        if not os.path.isfile(fileid_audio):
            raise RuntimeError(
                f"File {fileid_audio} doesn't exist"
            )
        waveform, sample_rate = torchaudio.load(fileid_audio)

        return (
            waveform,
            sample_rate,
            label,
            fileid
        )


    def __len__(self) -> int:
        return len(self._flist)
    