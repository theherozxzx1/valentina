"""Utilities for loading annotated datasets for Valentina training workflows."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, MutableMapping, Optional, Sequence

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency at runtime
    Image = None  # type: ignore[assignment]

try:  # pragma: no cover - torch is optional for lightweight installs
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - torch is optional for lightweight installs
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[misc, assignment]

__all__ = [
    "DatasetConfig",
    "DatasetRecord",
    "DatasetSplits",
    "load_annotated_dataset",
    "PromptImageDataset",
]


@dataclass(slots=True)
class DatasetRecord:
    """In-memory representation of a single annotated example."""

    image_path: Path
    prompt: str
    metadata: MutableMapping[str, Any] | None = None


@dataclass(slots=True)
class DatasetSplits:
    """Container for dataset splits used by the trainer."""

    train: List[DatasetRecord]
    validation: List[DatasetRecord] | None = None


@dataclass(slots=True)
class DatasetConfig:
    """Configuration block describing how to load and prepare annotated data."""

    dataset_root: Path
    annotations_file: Path
    resolution: int = 512
    shuffle: bool = True
    validation_split: float | None = None
    seed: int = 42
    image_column: str = "image"
    prompt_column: str = "prompt"
    metadata_column: str | None = "metadata"
    transform_factory: Callable[[int], Callable[["Image.Image"], Any]] | None = None
    extra_metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def resolve_paths(self) -> None:
        """Expand user variables and convert paths to their absolute form."""

        self.dataset_root = self.dataset_root.expanduser().resolve()
        self.annotations_file = (self.dataset_root / self.annotations_file).resolve()


def _open_annotations(path: Path) -> Iterable[MutableMapping[str, Any]]:
    """Open an annotation file supporting JSON, JSONL and CSV formats."""

    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        with path.open("r", encoding="utf8") as file:
            if suffix == ".jsonl":
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)
            else:
                payload = json.load(file)
                if isinstance(payload, list):
                    for record in payload:
                        if isinstance(record, dict):
                            yield record
                        else:  # pragma: no cover - guard for malformed files
                            raise ValueError("Invalid record in annotations JSON: expected dict")
                else:  # pragma: no cover - guard for malformed files
                    raise ValueError("Annotations JSON must contain a list of records")
    elif suffix in {".csv", ".tsv"}:
        dialect = "excel" if suffix == ".csv" else "excel-tab"
        with path.open("r", encoding="utf8", newline="") as file:
            reader = csv.DictReader(file, dialect=dialect)
            yield from reader
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported annotation format: {path.suffix}")


def _make_record(
    payload: MutableMapping[str, Any],
    dataset_root: Path,
    config: DatasetConfig,
) -> DatasetRecord:
    """Convert a raw annotation dictionary into a :class:`DatasetRecord`."""

    image_value = payload.get(config.image_column)
    prompt_value = payload.get(config.prompt_column)

    if image_value is None or prompt_value is None:
        raise ValueError(
            "Annotation record is missing required columns "
            f"'{config.image_column}' or '{config.prompt_column}'."
        )

    image_path = dataset_root / Path(str(image_value))
    metadata: MutableMapping[str, Any] | None = None
    if config.metadata_column and config.metadata_column in payload:
        raw_metadata = payload[config.metadata_column]
        if isinstance(raw_metadata, MutableMapping):
            metadata = dict(raw_metadata)
        elif raw_metadata is not None:  # pragma: no cover - tolerant branch
            raise ValueError(
                "Metadata column must contain a JSON object when provided in annotations."
            )

    return DatasetRecord(image_path=image_path, prompt=str(prompt_value), metadata=metadata)


def load_annotated_dataset(config: DatasetConfig) -> DatasetSplits:
    """Load an annotated dataset using the specification defined in ``config``.

    The loader is intentionally format-agnostic, supporting JSON/JSONL/CSV annotation
    files that contain at least two columns: one for the image path and one for the
    corresponding prompt/caption. Additional metadata columns are optional.
    """

    config.resolve_paths()

    if not config.annotations_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {config.annotations_file}")

    raw_records = [_make_record(payload, config.dataset_root, config) for payload in _open_annotations(config.annotations_file)]

    if not raw_records:
        raise ValueError(
            f"No annotated examples found in {config.annotations_file}."
        )

    if config.shuffle:
        random.Random(config.seed).shuffle(raw_records)

    validation: List[DatasetRecord] | None = None
    if config.validation_split:
        if not 0.0 < config.validation_split < 1.0:
            raise ValueError("validation_split must be between 0 and 1 when provided")
        split_index = int(len(raw_records) * (1 - config.validation_split))
        validation = raw_records[split_index:]
        raw_records = raw_records[:split_index]

    return DatasetSplits(train=raw_records, validation=validation)


class PromptImageDataset(Dataset):  # type: ignore[misc]
    """Torch dataset that yields image tensors and prompts for fine-tuning."""

    def __init__(
        self,
        records: Sequence[DatasetRecord],
        resolution: int,
        transform_factory: Callable[[int], Callable[["Image.Image"], Any]] | None = None,
    ) -> None:
        if torch is None or Image is None:
            raise ImportError(
                "PromptImageDataset requires both torch and pillow to be installed."
            )
        self._records = list(records)
        self._resolution = resolution
        self._transform = (
            transform_factory(resolution)
            if transform_factory is not None
            else self._default_transform(resolution)
        )

    @staticmethod
    def _default_transform(resolution: int) -> Callable[["Image.Image"], Any]:
        from torchvision import transforms  # local import to avoid hard dependency

        return transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> MutableMapping[str, Any]:
        record = self._records[index]
        image = Image.open(record.image_path).convert("RGB")
        pixel_values = self._transform(image)
        return {
            "pixel_values": pixel_values,
            "prompt": record.prompt,
            "metadata": record.metadata or {},
        }


def iter_records(config: DatasetConfig) -> Iterator[DatasetRecord]:
    """Yield :class:`DatasetRecord` items without materialising the entire dataset."""

    config.resolve_paths()
    for payload in _open_annotations(config.annotations_file):
        yield _make_record(payload, config.dataset_root, config)
