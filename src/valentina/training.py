"""Fine-tuning utilities built on top of ``diffusers`` for the Valentina toolkit."""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Protocol, Sequence

from .data import DatasetConfig, DatasetSplits, PromptImageDataset, load_annotated_dataset

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional heavy dependency
    import torch
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - optional heavy dependency
    torch = None  # type: ignore[assignment]
    DataLoader = object  # type: ignore[misc, assignment]

try:  # pragma: no cover - optional dependency
    from accelerate import Accelerator
except ImportError:  # pragma: no cover - optional dependency
    Accelerator = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from diffusers import DiffusionPipeline
except ImportError:  # pragma: no cover - optional dependency
    DiffusionPipeline = None  # type: ignore[assignment]

__all__ = [
    "DiffusersTrainingConfig",
    "ModelConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "ExperimentLogger",
    "EvaluationHook",
    "run_fine_tuning",
]


@dataclass(slots=True)
class ModelConfig:
    """Configuration block for the base model used during fine-tuning."""

    base_model: str
    revision: Optional[str] = None
    variant: Optional[str] = None
    text_encoder: Optional[str] = None
    token: Optional[str] = None


@dataclass(slots=True)
class OptimizerConfig:
    """Optimizer related settings for fine-tuning."""

    name: str = "adamw"
    learning_rate: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    eps: float = 1e-8


@dataclass(slots=True)
class SchedulerConfig:
    """Learning rate schedule used during training."""

    name: str = "cosine"
    warmup_steps: int = 500
    num_cycles: int = 1


@dataclass(slots=True)
class DiffusersTrainingConfig:
    """Hyper-parameters and training options for fine-tuning."""

    output_dir: Path
    model: ModelConfig
    train_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation: int = 1
    max_train_steps: int | None = None
    num_epochs: int = 1
    mixed_precision: str | None = "fp16"
    gradient_checkpointing: bool = True
    seed: int = 42
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    log_every: int = 10
    eval_every: int | None = 1000
    checkpointing_steps: int | None = 1000
    push_to_hub: bool = False

    def prepare(self) -> None:
        self.output_dir = self.output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class TrainingState:
    """Lightweight container shared with evaluation/logging callbacks."""

    epoch: int
    global_step: int
    num_train_examples: int
    losses: List[float] = field(default_factory=list)

    @property
    def mean_loss(self) -> float:
        if not self.losses:
            return float("nan")
        return float(sum(self.losses) / len(self.losses))


@dataclass(slots=True)
class EvaluationContext:
    """Data passed to evaluation hooks during training."""

    pipeline: Any
    state: TrainingState
    dataset_config: DatasetConfig
    outputs_dir: Path
    loggers: Sequence["ExperimentLogger"]


class ExperimentLogger(Protocol):
    """Protocol describing an experiment logger interface."""

    def start(self, config: DiffusersTrainingConfig, dataset: DatasetConfig) -> None:  # pragma: no cover - interface method
        ...

    def log(self, state: TrainingState, metrics: Mapping[str, float]) -> None:  # pragma: no cover - interface method
        ...

    def finish(self) -> None:  # pragma: no cover - interface method
        ...


class EvaluationHook(Protocol):
    """Protocol describing evaluation hooks executed after each epoch."""

    def __call__(self, context: EvaluationContext) -> None:  # pragma: no cover - interface method
        ...


def _build_dataloader(
    splits: DatasetSplits,
    dataset_config: DatasetConfig,
    training_config: DiffusersTrainingConfig,
) -> tuple[Any, Any | None]:
    if torch is None:
        raise ImportError("PyTorch must be installed to create the training dataloader.")

    dataset = PromptImageDataset(
        splits.train,
        resolution=dataset_config.resolution,
        transform_factory=dataset_config.transform_factory,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=training_config.train_batch_size,
        shuffle=True,
        drop_last=True,
    )

    validation_loader = None
    if splits.validation:
        validation_dataset = PromptImageDataset(
            splits.validation,
            resolution=dataset_config.resolution,
            transform_factory=dataset_config.transform_factory,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=training_config.eval_batch_size,
            shuffle=False,
        )

    return train_loader, validation_loader


def _prepare_pipeline(model_config: ModelConfig, mixed_precision: str | None) -> Any:
    if DiffusionPipeline is None:
        raise ImportError(
            "diffusers must be installed to instantiate the fine-tuning pipeline."
        )

    logger.info("Loading base pipeline %s", model_config.base_model)
    pipeline = DiffusionPipeline.from_pretrained(
        model_config.base_model,
        revision=model_config.revision,
        variant=model_config.variant,
        use_auth_token=model_config.token,
        torch_dtype=_resolve_dtype(mixed_precision),
    )
    pipeline.requires_safety_checker = False
    if hasattr(pipeline, "safety_checker"):
        pipeline.safety_checker = None
    pipeline.enable_attention_slicing()
    return pipeline


def _resolve_dtype(mixed_precision: str | None) -> Any:
    if torch is None:
        raise ImportError("PyTorch is required to configure tensor precision.")

    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":  # pragma: no cover - hardware dependant
        return torch.bfloat16
    return torch.float32


def _configure_optimizer(
    pipeline: Any,
    training_config: DiffusersTrainingConfig,
) -> tuple[Any, Any]:
    if torch is None:
        raise ImportError("PyTorch must be installed to build the optimizer.")

    params_to_optimize = pipeline.unet.parameters()

    if training_config.optimizer.name.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=training_config.optimizer.learning_rate,
            betas=training_config.optimizer.betas,
            eps=training_config.optimizer.eps,
            weight_decay=training_config.optimizer.weight_decay,
        )
    else:  # pragma: no cover - allow plugging different optimizers
        raise ValueError(
            f"Unsupported optimizer: {training_config.optimizer.name}."
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, training_config.max_train_steps or training_config.num_epochs),
        eta_min=0.0,
    )
    return optimizer, scheduler


def _run_logging_hooks(
    loggers: Sequence[ExperimentLogger],
    state: TrainingState,
    metrics: Mapping[str, float],
) -> None:
    for logger_instance in loggers:
        with contextlib.suppress(Exception):  # pragma: no cover - defensive
            logger_instance.log(state, metrics)


def _finalize_loggers(loggers: Sequence[ExperimentLogger]) -> None:
    for logger_instance in loggers:
        with contextlib.suppress(Exception):  # pragma: no cover - defensive
            logger_instance.finish()


def run_fine_tuning(
    dataset_config: DatasetConfig,
    training_config: DiffusersTrainingConfig,
    *,
    evaluation_hooks: Sequence[EvaluationHook] | None = None,
    loggers: Sequence[ExperimentLogger] | None = None,
) -> None:
    """Execute a full fine-tuning run following the high-level plan."""

    training_config.prepare()
    if torch is not None:
        torch.manual_seed(training_config.seed)
    splits = load_annotated_dataset(dataset_config)
    if splits.validation:
        dataset_config.extra_metadata.setdefault("validation_records", list(splits.validation))
    train_loader, validation_loader = _build_dataloader(splits, dataset_config, training_config)
    pipeline = _prepare_pipeline(training_config.model, training_config.mixed_precision)

    if Accelerator is None:
        raise ImportError(
            "accelerate must be installed to run distributed fine-tuning."
        )

    accelerator = Accelerator(gradient_accumulation_steps=training_config.gradient_accumulation)
    optimizer, scheduler = _configure_optimizer(pipeline, training_config)

    if hasattr(pipeline.unet, "enable_gradient_checkpointing") and training_config.gradient_checkpointing:
        pipeline.unet.enable_gradient_checkpointing()

    if hasattr(pipeline.text_encoder, "requires_grad_"):
        pipeline.text_encoder.requires_grad_(False)
    if hasattr(pipeline.vae, "requires_grad_"):
        pipeline.vae.requires_grad_(False)

    pipeline.text_encoder.to(accelerator.device)
    pipeline.vae.to(accelerator.device)

    pipeline.unet, optimizer, train_loader, scheduler = accelerator.prepare(
        pipeline.unet, optimizer, train_loader, scheduler
    )

    global_step = 0
    total_batch_size = (
        training_config.train_batch_size
        * training_config.gradient_accumulation
        * accelerator.num_processes
    )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_loader.dataset))  # type: ignore[arg-type]
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_batch_size)
    logger.info("  Gradient Accumulation steps = %d", training_config.gradient_accumulation)

    state = TrainingState(epoch=0, global_step=0, num_train_examples=len(train_loader.dataset))  # type: ignore[arg-type]

    if loggers:
        for logger_instance in loggers:
            with contextlib.suppress(Exception):  # pragma: no cover - defensive
                logger_instance.start(training_config, dataset_config)

    for epoch in range(training_config.num_epochs):
        pipeline.train()
        state.epoch = epoch
        state.losses.clear()

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(pipeline.unet):
                pixel_values = batch["pixel_values"].to(
                    accelerator.device, dtype=pipeline.unet.dtype
                )
                latents = pipeline.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * pipeline.vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    pipeline.scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                )
                noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

                text_inputs = pipeline.tokenizer(
                    batch["prompt"],
                    padding="max_length",
                    max_length=pipeline.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(accelerator.device)
                encoder_hidden_states = pipeline.text_encoder(text_input_ids)[0]
                noise_pred = pipeline.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                target = noise
                loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(pipeline.unet.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            state.global_step = global_step
            state.losses.append(loss.detach().item())

            if training_config.log_every and global_step % training_config.log_every == 0:
                metrics = {"loss": state.losses[-1], "lr": optimizer.param_groups[0]["lr"]}
                _run_logging_hooks(loggers or (), state, metrics)
                logger.info("Step %d - loss: %.4f - lr: %.6f", global_step, metrics["loss"], metrics["lr"])

            if training_config.checkpointing_steps and global_step % training_config.checkpointing_steps == 0:
                save_path = training_config.output_dir / f"checkpoint-{global_step}"
                accelerator.wait_for_everyone()
                pipeline.save_pretrained(save_path)
                logger.info("Saved checkpoint to %s", save_path)

            if training_config.max_train_steps and global_step >= training_config.max_train_steps:
                break

        if training_config.eval_every and evaluation_hooks and (epoch + 1) % 1 == 0:
            evaluation_context = EvaluationContext(
                pipeline=pipeline,
                state=state,
                dataset_config=dataset_config,
                outputs_dir=training_config.output_dir,
                loggers=loggers or (),
            )
            for hook in evaluation_hooks:
                with contextlib.suppress(Exception):  # pragma: no cover - defensive
                    hook(evaluation_context)

        if training_config.max_train_steps and global_step >= training_config.max_train_steps:
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()

    pipeline.save_pretrained(training_config.output_dir)
    logger.info("Training complete. Model saved to %s", training_config.output_dir)

    _finalize_loggers(loggers or ())


def create_clip_similarity_hook(model_name: str = "ViT-B/32") -> EvaluationHook:
    """Factory for an evaluation hook that computes CLIP similarity."""

    def _hook(context: EvaluationContext) -> None:
        try:
            import clip  # type: ignore[import]
        except ImportError as error:  # pragma: no cover - optional dependency
            logger.warning("CLIP evaluation skipped: %s", error)
            return

        clip_model, preprocess = clip.load(model_name, device=context.pipeline.device)
        prompts = [record.prompt for record in context.dataset_config.extra_metadata.get("validation_records", [])]
        if not prompts and context.dataset_config.validation_split:
            prompts = [""] * len(prompts)
        if not prompts:
            logger.warning("CLIP evaluation skipped: validation prompts not available")
            return

        images = []
        for record in context.dataset_config.extra_metadata.get("validation_records", []):
            try:
                from PIL import Image

                images.append(preprocess(Image.open(record.image_path)).unsqueeze(0))
            except Exception as error:  # pragma: no cover - defensive
                logger.warning("Failed to load image %s: %s", record.image_path, error)
        if not images:
            return

        import torch

        image_input = torch.cat(images).to(context.pipeline.device)
        text_tokens = clip.tokenize(prompts).to(context.pipeline.device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_tokens)
            similarity = torch.nn.functional.cosine_similarity(
                image_features, text_features, dim=-1
            )

        avg_similarity = similarity.mean().item()
        logger.info("CLIP similarity: %.4f", avg_similarity)
        _run_logging_hooks(
            context.loggers,
            context.state,
            {"clip_similarity": avg_similarity},
        )

    return _hook


def create_face_embedding_hook(model_name: str = "buffalo_l") -> EvaluationHook:
    """Factory for an evaluation hook that computes face embedding distances."""

    def _hook(context: EvaluationContext) -> None:
        try:
            from insightface.app import FaceAnalysis  # type: ignore[import]
        except ImportError as error:  # pragma: no cover - optional dependency
            logger.warning("Face embedding evaluation skipped: %s", error)
            return

        app = FaceAnalysis(name=model_name)
        app.prepare(ctx_id=0, det_size=(256, 256))

        records = context.dataset_config.extra_metadata.get("validation_records", [])
        if not records:
            logger.warning("Face embedding evaluation skipped: validation records missing")
            return

        embeddings: List[Any] = []
        for record in records:
            try:
                from PIL import Image

                image = Image.open(record.image_path).convert("RGB")
                faces = app.get(image)
                if faces:
                    embeddings.append(faces[0].embedding)
            except Exception as error:  # pragma: no cover - defensive
                logger.warning("Failed to process image %s: %s", record.image_path, error)

        if len(embeddings) < 2:
            logger.warning("Not enough embeddings to compute similarity")
            return

        import numpy as np

        ref = embeddings[0]
        similarities = [float(np.dot(ref, emb) / (np.linalg.norm(ref) * np.linalg.norm(emb))) for emb in embeddings[1:]]
        avg_similarity = sum(similarities) / len(similarities)
        logger.info("Face embedding similarity: %.4f", avg_similarity)
        _run_logging_hooks(
            context.loggers,
            context.state,
            {"face_embedding_similarity": avg_similarity},
        )

    return _hook


class WandbLogger:
    """Experiment logger backed by Weights & Biases."""

    def __init__(self, project: str, run_name: str | None = None) -> None:
        self._project = project
        self._run_name = run_name
        self._run = None

    def start(self, config: DiffusersTrainingConfig, dataset: DatasetConfig) -> None:
        import wandb

        self._run = wandb.init(project=self._project, name=self._run_name, config={
            "training_config": dataclasses.asdict(config),
            "dataset_config": dataclasses.asdict(dataset),
        })

    def log(self, state: TrainingState, metrics: Mapping[str, float]) -> None:
        if self._run is None:
            return
        self._run.log({"step": state.global_step, **metrics})

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None


class MLflowLogger:
    """Experiment logger backed by MLflow."""

    def __init__(self, experiment: str | None = None, run_name: str | None = None) -> None:
        self._experiment = experiment
        self._run_name = run_name
        self._active_run = None

    def start(self, config: DiffusersTrainingConfig, dataset: DatasetConfig) -> None:
        import mlflow

        if self._experiment is not None:
            mlflow.set_experiment(self._experiment)
        self._active_run = mlflow.start_run(run_name=self._run_name)
        mlflow.log_params({
            **{f"training.{k}": v for k, v in dataclasses.asdict(config).items()},
            **{f"dataset.{k}": v for k, v in dataclasses.asdict(dataset).items()},
        })

    def log(self, state: TrainingState, metrics: Mapping[str, float]) -> None:
        if self._active_run is None:
            return
        import mlflow

        mlflow.log_metrics(metrics, step=state.global_step)

    def finish(self) -> None:
        if self._active_run is not None:
            import mlflow

            mlflow.end_run()
            self._active_run = None


def build_logger(name: str, **kwargs: Any) -> ExperimentLogger:
    name = name.lower()
    if name == "wandb":
        return WandbLogger(**kwargs)
    if name == "mlflow":
        return MLflowLogger(**kwargs)
    raise ValueError(f"Unknown logger backend: {name}")
