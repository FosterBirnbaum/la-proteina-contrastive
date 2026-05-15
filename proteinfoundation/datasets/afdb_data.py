"""AFDB dataset module mirroring `pdb_data.py`.

Pipeline assumes structures have already been fetched with
`script_utils/download_afdb.py` into `<data_dir>/raw/AF-<uniprot>-F1-model_v<version>.pdb`.
The author-released zip ships a single flat ID list per subset (no train/val/
test partitioning), so ``AFDBDataSplitter`` always generates a fresh random
split. See ``docs/AFDB.md`` and paper App. C.1 for context.
"""

from __future__ import annotations

import pathlib
import re
from typing import Callable, List, Literal, Optional

import pandas as pd
from loguru import logger

from proteinfoundation.datasets.pdb_data import (
    PDBDataSplitter,
    PDBLightningDataModule,
)


_AF_NAME_RE = re.compile(r"^AF-([A-Za-z0-9]+)-F\d+-model_v\d+$")


def _normalise_to_file_stem(token: str, version: int) -> str:
    """Accept either a bare UniProt accession or a full AF-...-model_vN stem."""
    token = token.strip()
    if not token:
        return ""
    if token.startswith("AF-"):
        return token[:-4] if token.endswith(".pdb") else token
    return f"AF-{token}-F1-model_v{version}"


def _read_id_file(path: pathlib.Path, version: int) -> List[str]:
    stems: List[str] = []
    with path.open() as f:
        for line in f:
            tok = line.strip().split()[0] if line.strip() else ""
            if not tok or tok.startswith("#"):
                continue
            stems.append(_normalise_to_file_stem(tok, version))
    return stems


class AFDBDataSelector:
    """Build a dataframe from a La-Proteina AFDB ID list.

    The author-released zip (``laproteina_afdb_ids.zip``) contains two files:
    ``AFDB_IDs-512.txt`` (the ~344k Foldseek-clustered subset used for the
    LD1/LD2 models, lengths 32-512) and ``AFDB_IDs-896.txt`` (the 46.9M
    long-protein subset used for LD3 only). Pick one via ``ids_file``.

    The zip does **not** ship train/val/test splits. ``AFDBDataSplitter`` will
    generate a random split (defaulting to ~99.9% / 0.1% / 0%, matching the
    model card) from whatever the selector returns.

    Args:
        data_dir: Root directory; raw PDBs are expected under ``data_dir/raw``.
        ids_file: Path to a single newline-delimited ID file. Accepts either
            bare UniProt accessions or ``AF-<uniprot>-F1-model_v<N>`` stems.
        fraction: Optional subsampling for quick experiments. Defaults to 1.0.
        version: AFDB model version (default 4).
    """

    def __init__(
        self,
        data_dir: str,
        ids_file: str,
        fraction: float = 1.0,
        version: int = 4,
    ) -> None:
        self.database = "afdb"
        self.data_dir = pathlib.Path(data_dir)
        self.ids_file = pathlib.Path(ids_file)
        self.fraction = fraction
        self.version = version
        self.df_data: Optional[pd.DataFrame] = None

    def create_dataset(self) -> pd.DataFrame:
        if self.df_data is not None:
            return self.df_data
        if not self.ids_file.exists():
            raise FileNotFoundError(f"AFDB ids_file does not exist: {self.ids_file}")

        stems = _read_id_file(self.ids_file, self.version)
        logger.info(f"AFDBDataSelector: read {len(stems)} IDs from {self.ids_file.name}")
        rows = [{"pdb": s, "id": s} for s in stems]

        if not rows:
            raise RuntimeError(f"No AFDB IDs read from {self.ids_file}")

        df = pd.DataFrame(rows).drop_duplicates(subset=["pdb"]).reset_index(drop=True)
        if self.fraction != 1.0:
            logger.info(f"Subsampling to fraction={self.fraction}")
            df = df.sample(frac=self.fraction, random_state=42).reset_index(drop=True)
        logger.info(f"AFDBDataSelector: {len(df)} entries")
        self.df_data = df
        return df


class AFDBDataSplitter(PDBDataSplitter):
    """Random train/val/test splitter for AFDB.

    The authors did not publish an explicit split for the AFDB ID lists. The
    model card describes a 99.9% / 0.1% train/val ratio (no test split) but
    the exact 1,300-structure validation set used during training is not
    enumerated. We therefore generate a fresh random split with a fixed seed.

    Forces ``split_type="random"``: AFDB-512 is already Foldseek-clustered at
    the source, so re-clustering with MMseqs2 is unnecessary and very expensive
    on 344k samples.
    """

    def split_data(self, df_data: pd.DataFrame, file_identifier: str):
        if self.split_type != "random":
            logger.warning(
                f"AFDBDataSplitter: overriding split_type={self.split_type!r} -> 'random' "
                "(AFDB-512 is already Foldseek-clustered)."
            )
            self.split_type = "random"
        return super().split_data(df_data, file_identifier)


class AFDBLightningDataModule(PDBLightningDataModule):
    """Datamodule for AFDB structures pre-downloaded by ``download_afdb.py``."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        dataselector: Optional[AFDBDataSelector] = None,
        datasplitter: Optional[AFDBDataSplitter] = None,
        in_memory: bool = False,
        format: Literal["pdb"] = "pdb",
        overwrite: bool = False,
        store_het: bool = False,
        store_bfactor: bool = True,
        batch_padding: bool = True,
        sampling_mode: Literal["random", "cluster-random", "cluster-reps"] = "random",
        transforms: Optional[List[Callable]] = None,
        pre_transforms: Optional[List[Callable]] = None,
        pre_filters: Optional[List[Callable]] = None,
        batch_size: int = 32,
        num_workers: int = 32,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            dataselector=dataselector,
            datasplitter=datasplitter,
            in_memory=in_memory,
            format=format,
            overwrite=overwrite,
            store_het=store_het,
            store_bfactor=store_bfactor,
            batch_padding=batch_padding,
            sampling_mode=sampling_mode,
            transforms=transforms,
            pre_transforms=pre_transforms,
            pre_filters=pre_filters,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs,
        )

    def _get_file_identifier(self, ds: AFDBDataSelector) -> str:
        return f"df_afdb_f{ds.fraction}_v{ds.version}_{ds.ids_file.stem}"

    def _download_structure_data(self, pdb_codes) -> None:
        """AFDB structures are pre-downloaded by ``script_utils/download_afdb.py``.

        We only verify that files exist and warn (but do not error) when any are
        missing -- those entries will be skipped by ``_load_and_process_pdb``.
        """
        if not pdb_codes:
            return
        missing = [
            p for p in pdb_codes
            if not (self.raw_dir / f"{p}.{self.format}").exists()
            and not (self.raw_dir / f"{p}.{self.format}.gz").exists()
        ]
        if missing:
            logger.warning(
                f"{len(missing)} / {len(pdb_codes)} AFDB structures missing from {self.raw_dir}. "
                f"Run script_utils/download_afdb.py to fetch them. "
                f"Examples: {missing[:5]}"
            )
        else:
            logger.info(f"All {len(pdb_codes)} AFDB structures present in {self.raw_dir}.")

    def prepare_data(self):
        # We always go through the dataselector branch; there is no graphein PDBManager for AFDB.
        if self.dataselector is None:
            raise ValueError("AFDBLightningDataModule requires a dataselector.")
        file_identifier = self._get_file_identifier(self.dataselector)
        df_data_name = f"{file_identifier}.csv"
        if not self.overwrite and (self.data_dir / df_data_name).exists():
            logger.info(
                f"{df_data_name} already exists, skipping data selection and processing stage."
            )
            return
        logger.info(f"{df_data_name} does not exist yet, creating dataset now.")
        df_data = self.dataselector.create_dataset()
        logger.info(f"Dataset created with {len(df_data)} entries.")
        self._download_structure_data(df_data["pdb"].tolist())
        self._process_structure_data(df_data["pdb"].tolist(), chains=None)
        logger.info(f"Saving dataset csv to {df_data_name}")
        df_data.to_csv(self.data_dir / df_data_name, index=False)
