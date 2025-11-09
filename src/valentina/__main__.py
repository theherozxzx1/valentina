"""Command line entry point for the Valentina toolkit."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from . import __version__
from .data import DatasetConfig
from .modal import ModalConfig, StorageConfig, launch_modal_training
from .training import (
    DiffusersTrainingConfig,
    EvaluationHook,
    ExperimentLogger,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    build_logger,
    create_clip_similarity_hook,
    create_face_embedding_hook,
    run_fine_tuning,
)

logger = logging.getLogger(__name__)


def _collect_evaluation_hooks(args: argparse.Namespace) -> list[EvaluationHook]:
    hooks: list[EvaluationHook] = []
    if args.eval_clip:
        hooks.append(create_clip_similarity_hook())
    if args.eval_face:
        hooks.append(create_face_embedding_hook())
    return hooks


def _collect_loggers(args: argparse.Namespace) -> list[ExperimentLogger]:
    loggers: list[ExperimentLogger] = []
    if not args.logger:
        return loggers
    for name in args.logger:
        if name == "wandb":
            project = args.wandb_project or "valentina"
            loggers.append(build_logger("wandb", project=project, run_name=args.wandb_run))
        elif name == "mlflow":
            loggers.append(
                build_logger(
                    "mlflow",
                    experiment=args.mlflow_experiment,
                    run_name=args.mlflow_run,
                )
            )
        else:  # pragma: no cover - argparse should prevent this branch
            raise ValueError(f"Unsupported logger '{name}'")
    return loggers


def _build_dataset_config(args: argparse.Namespace) -> DatasetConfig:
    return DatasetConfig(
        dataset_root=Path(args.dataset_root),
        annotations_file=Path(args.annotations_file),
        resolution=args.resolution,
        validation_split=args.validation_split,
        seed=args.seed,
        image_column=args.image_column,
        prompt_column=args.prompt_column,
        metadata_column=args.metadata_column,
    )


def _build_training_config(args: argparse.Namespace) -> DiffusersTrainingConfig:
    optimizer = OptimizerConfig(
        learning_rate=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.optimizer_eps,
    )
    scheduler = SchedulerConfig(
        name=args.scheduler,
        warmup_steps=args.warmup_steps,
        num_cycles=args.scheduler_cycles,
    )
    mixed_precision = None if args.mixed_precision == "no" else args.mixed_precision
    return DiffusersTrainingConfig(
        output_dir=Path(args.output_dir),
        model=ModelConfig(
            base_model=args.base_model,
            revision=args.model_revision,
            variant=args.model_variant,
            token=args.hf_token,
        ),
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation=args.gradient_accumulation,
        max_train_steps=args.max_train_steps,
        num_epochs=args.epochs,
        mixed_precision=mixed_precision,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        seed=args.seed,
        optimizer=optimizer,
        scheduler=scheduler,
        log_every=args.log_every,
        eval_every=args.eval_every,
        checkpointing_steps=args.checkpointing_steps,
        push_to_hub=args.push_to_hub,
    )


def _maybe_configure_storage(args: argparse.Namespace) -> StorageConfig | None:
    if not args.storage_provider or not args.storage_bucket:
        return None
    return StorageConfig(
        provider=args.storage_provider,
        bucket=args.storage_bucket,
        prefix=args.storage_prefix or "",
        mount_path=Path(args.storage_mount_path),
        secret_name=args.storage_secret,
        upload_after_training=not args.no_storage_upload,
    )


def _dispatch_training(args: argparse.Namespace) -> None:
    dataset_config = _build_dataset_config(args)
    training_config = _build_training_config(args)
    evaluation_hooks = _collect_evaluation_hooks(args)
    loggers = _collect_loggers(args)

    if args.use_modal:
        try:
            modal_config = ModalConfig(
                stub_name=args.modal_stub_name,
                function_name="run_fine_tuning",
                python_version=args.modal_python,
                gpu=args.modal_gpu,
                timeout=args.modal_timeout,
                secrets=tuple(args.modal_secret or ()),
            )
            if args.modal_package:
                modal_config.pip_packages = tuple(args.modal_package)
            storage = _maybe_configure_storage(args)
            launch_modal_training(
                dataset_config,
                training_config,
                modal_config,
                evaluation_hooks=evaluation_hooks,
                loggers=loggers,
                storage=storage,
                wait=not args.modal_async,
            )
        except Exception as error:  # pragma: no cover - runtime dependent
            logger.error("Failed to launch Modal training: %s", error)
            raise
    else:
        run_fine_tuning(
            dataset_config,
            training_config,
            evaluation_hooks=evaluation_hooks,
            loggers=loggers,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="valentina",
        description=(
            "Utilities for working with the Valentina Moreau generative image workflows."
        ),
    )
    parser.add_argument("--version", action="version", version=f"valentina {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Fine-tune a diffusion model with annotated data.")
    train_parser.add_argument("--dataset-root", required=True, help="Root directory containing the dataset assets.")
    train_parser.add_argument("--annotations-file", required=True, help="Relative path to the annotations file.")
    train_parser.add_argument("--image-column", default="image", help="Column name with the image relative path.")
    train_parser.add_argument("--prompt-column", default="prompt", help="Column name with the text prompt.")
    train_parser.add_argument("--metadata-column", default="metadata", help="Column name with optional metadata.")
    train_parser.add_argument("--resolution", type=int, default=512, help="Target training resolution in pixels.")
    train_parser.add_argument("--validation-split", type=float, default=None, help="Proportion of data used for validation.")
    train_parser.add_argument("--base-model", required=True, help="Identifier of the base diffusers model.")
    train_parser.add_argument("--model-revision", default=None, help="Optional model revision to pull from the hub.")
    train_parser.add_argument("--model-variant", default=None, help="Optional model variant for specialized checkpoints.")
    train_parser.add_argument("--hf-token", default=None, help="Authentication token for private models on Hugging Face.")
    train_parser.add_argument("--output-dir", required=True, help="Where to store fine-tuning checkpoints and artifacts.")
    train_parser.add_argument("--train-batch-size", type=int, default=1)
    train_parser.add_argument("--eval-batch-size", type=int, default=1)
    train_parser.add_argument("--gradient-accumulation", type=int, default=1)
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--max-train-steps", type=int, default=None)
    train_parser.add_argument(
        "--mixed-precision",
        choices=["no", "fp16", "bf16"],
        default="fp16",
        help="Precision strategy used during fine-tuning.",
    )
    train_parser.add_argument("--learning-rate", type=float, default=1e-5)
    train_parser.add_argument("--beta1", type=float, default=0.9)
    train_parser.add_argument("--beta2", type=float, default=0.999)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--optimizer-eps", type=float, default=1e-8)
    train_parser.add_argument("--scheduler", default="cosine", help="Scheduler strategy to employ.")
    train_parser.add_argument("--scheduler-cycles", type=int, default=1)
    train_parser.add_argument("--warmup-steps", type=int, default=500)
    train_parser.add_argument("--log-every", type=int, default=10)
    train_parser.add_argument("--eval-every", type=int, default=1000)
    train_parser.add_argument("--checkpointing-steps", type=int, default=1000)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--push-to-hub", action="store_true", help="Enable pushing checkpoints to the Hugging Face Hub.")
    train_parser.add_argument("--no-gradient-checkpointing", action="store_true", help="Disable gradient checkpointing.")
    train_parser.add_argument("--eval-clip", action="store_true", help="Enable CLIP similarity evaluation hook.")
    train_parser.add_argument("--eval-face", action="store_true", help="Enable face embedding evaluation hook.")
    train_parser.add_argument(
        "--logger",
        action="append",
        choices=["wandb", "mlflow"],
        help="Register one or more experiment loggers.",
    )
    train_parser.add_argument("--wandb-project", default=None)
    train_parser.add_argument("--wandb-run", default=None)
    train_parser.add_argument("--mlflow-experiment", default=None)
    train_parser.add_argument("--mlflow-run", default=None)
    train_parser.add_argument("--use-modal", action="store_true", help="Execute training remotely on Modal.")
    train_parser.add_argument("--modal-stub-name", default="valentina-trainer")
    train_parser.add_argument("--modal-python", default="3.10")
    train_parser.add_argument("--modal-gpu", default="A10G")
    train_parser.add_argument("--modal-timeout", type=int, default=3600)
    train_parser.add_argument("--modal-package", action="append", help="Additional pip packages to install on Modal.")
    train_parser.add_argument("--modal-secret", action="append", help="Modal secrets required for the job.")
    train_parser.add_argument("--modal-async", action="store_true", help="Return immediately instead of waiting for completion.")
    train_parser.add_argument(
        "--storage-provider",
        choices=["s3", "gcs"],
        default=None,
        help="Cloud storage provider for remote checkpoints.",
    )
    train_parser.add_argument("--storage-bucket", default=None, help="Bucket name for checkpoint uploads.")
    train_parser.add_argument("--storage-prefix", default="", help="Optional prefix within the storage bucket.")
    train_parser.add_argument(
        "--storage-mount-path",
        default="/checkpoints",
        help="Local mount path inside the Modal container for checkpoint storage.",
    )
    train_parser.add_argument("--storage-secret", default=None, help="Secret used to authenticate with the storage provider.")
    train_parser.add_argument("--no-storage-upload", action="store_true", help="Disable checkpoint uploads after remote runs.")

    train_parser.set_defaults(func=_dispatch_training)

    return parser


def cli(argv: Sequence[str] | None = None) -> None:
    """Run the command line interface."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return

    logging.basicConfig(level=logging.INFO)
    args.func(args)


if __name__ == "__main__":
    cli()
