"""Integration helpers for launching fine-tuning jobs on Modal."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

from .data import DatasetConfig
from .training import DiffusersTrainingConfig, EvaluationHook, ExperimentLogger, run_fine_tuning

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import modal
except ImportError:  # pragma: no cover - optional dependency
    modal = None  # type: ignore[assignment]


__all__ = [
    "ModalConfig",
    "StorageConfig",
    "build_modal_image",
    "register_modal_training_function",
    "launch_modal_training",
]


@dataclass(slots=True)
class StorageConfig:
    """Represents a remote storage target for checkpoints produced on Modal."""

    provider: Literal["s3", "gcs"]
    bucket: str
    prefix: str = ""
    mount_path: Path = Path("/checkpoints")
    secret_name: Optional[str] = None
    upload_after_training: bool = True

    def build_mount(self) -> Any:
        """Create a Modal mount for the configured storage provider if possible."""

        if modal is None:
            raise ImportError("Modal must be installed to create mounts.")

        mount_cls = getattr(modal, "Mount", None)
        if mount_cls is None:
            raise RuntimeError("modal.Mount is not available in the installed version of Modal")

        kwargs: dict[str, Any] = {
            "bucket": self.bucket,
            "remote_path": self.prefix or "/",
            "mount_path": str(self.mount_path),
        }

        if self.secret_name:
            secret_cls = getattr(modal, "Secret", None)
            if secret_cls is None:  # pragma: no cover - defensive
                raise RuntimeError("modal.Secret is not available to resolve storage credentials")
            kwargs["secret"] = secret_cls.from_name(self.secret_name)

        if self.provider == "s3":
            if hasattr(mount_cls, "from_s3"):
                return mount_cls.from_s3(**kwargs)
            raise RuntimeError("Installed Modal SDK does not expose Mount.from_s3")
        if self.provider == "gcs":
            if hasattr(mount_cls, "from_gcs"):
                return mount_cls.from_gcs(**kwargs)
            raise RuntimeError("Installed Modal SDK does not expose Mount.from_gcs")
        raise ValueError(f"Unsupported storage provider: {self.provider}")


@dataclass(slots=True)
class ModalConfig:
    """Encapsulates runtime configuration for Modal fine-tuning jobs."""

    stub_name: str = "valentina-trainer"
    function_name: str = "run_fine_tuning"
    python_version: str = "3.10"
    pip_packages: Sequence[str] = field(
        default_factory=lambda: (
            "torch", "diffusers[torch]", "accelerate", "safetensors", "transformers", "wandb", "mlflow"
        )
    )
    gpu: Optional[str] = "A10G"
    timeout: int = 60 * 60  # seconds
    secrets: Sequence[str] = field(default_factory=tuple)

    def resolve_gpu(self) -> Any:
        if modal is None or self.gpu is None:
            return None
        gpu_module = getattr(modal, "gpu", None)
        if gpu_module is None:
            raise RuntimeError("modal.gpu module is not available")
        if not hasattr(gpu_module, self.gpu):
            raise ValueError(f"GPU type '{self.gpu}' is not available in modal.gpu")
        gpu_cls = getattr(gpu_module, self.gpu)
        return gpu_cls()

    def resolve_secrets(self) -> list[Any] | None:
        if modal is None or not self.secrets:
            return None
        secret_cls = getattr(modal, "Secret", None)
        if secret_cls is None:
            raise RuntimeError("modal.Secret is not available")
        return [secret_cls.from_name(name) for name in self.secrets]


def build_modal_image(config: ModalConfig) -> Any:
    """Build the Modal image used for fine-tuning."""

    if modal is None:
        raise ImportError("Modal must be installed to build remote images.")

    image = modal.Image.debian_slim(python_version=config.python_version)
    image = image.pip_install(*config.pip_packages)
    return image


def _sync_checkpoints_to_storage(output_dir: Path, storage: StorageConfig) -> None:
    """Upload produced checkpoints to the configured storage backend."""

    if storage.provider == "s3":
        try:
            import boto3
        except ImportError as error:  # pragma: no cover - optional dependency
            logger.warning("Skipping S3 upload due to missing dependency: %s", error)
            return
        client = boto3.client("s3")
        for path in output_dir.rglob("*"):
            if path.is_file():
                relative_path = path.relative_to(output_dir)
                key = str(Path(storage.prefix) / relative_path)
                logger.info("Uploading %s to s3://%s/%s", path, storage.bucket, key)
                client.upload_file(str(path), storage.bucket, key)
        return
    if storage.provider == "gcs":
        try:
            from google.cloud import storage as gcs_storage  # type: ignore[import]
        except ImportError as error:  # pragma: no cover - optional dependency
            logger.warning("Skipping GCS upload due to missing dependency: %s", error)
            return
        client = gcs_storage.Client()
        bucket = client.bucket(storage.bucket)
        for path in output_dir.rglob("*"):
            if path.is_file():
                relative_path = path.relative_to(output_dir)
                blob = bucket.blob(str(Path(storage.prefix) / relative_path))
                logger.info("Uploading %s to gs://%s/%s", path, storage.bucket, blob.name)
                blob.upload_from_filename(str(path))
        return
    raise ValueError(f"Unsupported storage provider: {storage.provider}")


def register_modal_training_function(
    dataset_config: DatasetConfig,
    training_config: DiffusersTrainingConfig,
    modal_config: ModalConfig,
    *,
    evaluation_hooks: Sequence[EvaluationHook] | None = None,
    loggers: Sequence[ExperimentLogger] | None = None,
    storage: StorageConfig | None = None,
) -> Any:
    """Register a Modal function that executes :func:`run_fine_tuning`."""

    if modal is None:
        raise ImportError("Modal must be installed to register remote functions.")

    image = build_modal_image(modal_config)
    stub = modal.Stub(modal_config.stub_name)
    mounts = []
    if storage is not None:
        with contextlib.suppress(Exception):  # type: no cover - optional
            mounts.append(storage.build_mount())

    secrets = modal_config.resolve_secrets()

    @stub.function(
        image=image,
        gpu=modal_config.resolve_gpu(),
        timeout=modal_config.timeout,
        mounts=mounts or None,
        secrets=secrets,
    )
    def run_remote() -> None:
        logger.info("Starting remote fine-tuning via Modal")
        run_fine_tuning(
            dataset_config,
            training_config,
            evaluation_hooks=evaluation_hooks or (),
            loggers=loggers or (),
        )
        if storage and storage.upload_after_training:
            _sync_checkpoints_to_storage(training_config.output_dir, storage)

    return run_remote


def launch_modal_training(
    dataset_config: DatasetConfig,
    training_config: DiffusersTrainingConfig,
    modal_config: ModalConfig,
    *,
    evaluation_hooks: Sequence[EvaluationHook] | None = None,
    loggers: Sequence[ExperimentLogger] | None = None,
    storage: StorageConfig | None = None,
    wait: bool = True,
) -> None:
    """Convenience wrapper to register and execute the Modal training function."""

    if modal is None:
        raise ImportError("Modal must be installed to launch remote jobs.")

    function = register_modal_training_function(
        dataset_config,
        training_config,
        modal_config,
        evaluation_hooks=evaluation_hooks,
        loggers=loggers,
        storage=storage,
    )

    if wait:
        function.call()
    else:
        function.spawn()
