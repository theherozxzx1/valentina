"""Utilities for running text-to-image generation pipelines."""

from __future__ import annotations

import contextlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Mapping, MutableMapping, Optional, Sequence

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional heavy dependency
    import torch
except ImportError:  # pragma: no cover - optional heavy dependency
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional heavy dependency
    from diffusers import (
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        DiffusionPipeline,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        HeunDiscreteScheduler,
        LMSDiscreteScheduler,
        PNDMScheduler,
    )
except ImportError:  # pragma: no cover - optional heavy dependency
    DiffusionPipeline = None  # type: ignore[assignment]
    DDIMScheduler = None  # type: ignore[assignment]
    DPMSolverMultistepScheduler = None  # type: ignore[assignment]
    EulerDiscreteScheduler = None  # type: ignore[assignment]
    EulerAncestralDiscreteScheduler = None  # type: ignore[assignment]
    HeunDiscreteScheduler = None  # type: ignore[assignment]
    LMSDiscreteScheduler = None  # type: ignore[assignment]
    PNDMScheduler = None  # type: ignore[assignment]

__all__ = [
    "GenerationConfig",
    "build_generation_pipeline",
    "generate_images",
]


_SCHEDULER_ALIASES: Mapping[str, str] = {
    "ddim": "ddim",
    "dpm_solver": "dpm_solver",
    "dpm": "dpm_solver",
    "euler": "euler",
    "euler_a": "euler_a",
    "heun": "heun",
    "lms": "lms",
    "pndm": "pndm",
}


def _available_schedulers() -> MutableMapping[str, Any]:
    factories: MutableMapping[str, Any] = {}
    if DDIMScheduler is not None:
        factories["ddim"] = DDIMScheduler
    if DPMSolverMultistepScheduler is not None:
        factories["dpm_solver"] = DPMSolverMultistepScheduler
    if EulerDiscreteScheduler is not None:
        factories["euler"] = EulerDiscreteScheduler
    if EulerAncestralDiscreteScheduler is not None:
        factories["euler_a"] = EulerAncestralDiscreteScheduler
    if HeunDiscreteScheduler is not None:
        factories["heun"] = HeunDiscreteScheduler
    if LMSDiscreteScheduler is not None:
        factories["lms"] = LMSDiscreteScheduler
    if PNDMScheduler is not None:
        factories["pndm"] = PNDMScheduler
    return factories


def _resolve_dtype(mixed_precision: str | None) -> Any:
    if torch is None:
        raise ImportError("PyTorch must be installed to configure generation precision.")
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":  # pragma: no cover - hardware dependent
        return torch.bfloat16
    return torch.float32


def _sanitize_prompt(prompt: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\-_.]+", "_", prompt.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "prompt"


def _select_device(device: str | None) -> str:
    if device:
        return device
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass(slots=True)
class GenerationConfig:
    """Configuration options controlling text-to-image generation."""

    base_model: str
    output_dir: Path
    revision: Optional[str] = None
    variant: Optional[str] = None
    token: Optional[str] = None
    width: int = 512
    height: int = 512
    guidance_scale: float = 7.5
    num_inference_steps: int = 30
    scheduler: Optional[str] = None
    seed: Optional[int] = None
    images_per_prompt: int = 1
    negative_prompt: Optional[str] = None
    mixed_precision: Optional[str] = "fp16"
    lora_weights: Optional[Path] = None
    enable_xformers: bool = True
    attention_slicing: bool = True
    disable_safety_checker: bool = True
    device: Optional[str] = None

    extra_kwargs: MutableMapping[str, Any] = field(default_factory=dict)

    def prepare(self) -> None:
        self.output_dir = self.output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.lora_weights is not None:
            self.lora_weights = self.lora_weights.expanduser().resolve()


def build_generation_pipeline(config: GenerationConfig) -> Any:
    """Instantiate a ``diffusers`` pipeline following ``config``."""

    if DiffusionPipeline is None:
        raise ImportError("diffusers must be installed to run image generation.")

    dtype = _resolve_dtype(config.mixed_precision)
    pipeline = DiffusionPipeline.from_pretrained(
        config.base_model,
        revision=config.revision,
        variant=config.variant,
        torch_dtype=dtype,
        use_auth_token=config.token,
    )

    if config.disable_safety_checker and hasattr(pipeline, "safety_checker"):
        pipeline.safety_checker = None

    if config.attention_slicing and hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()

    if config.enable_xformers and hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
        with contextlib.suppress(Exception):  # pragma: no cover - optional acceleration
            pipeline.enable_xformers_memory_efficient_attention()

    if config.scheduler:
        alias = _SCHEDULER_ALIASES.get(config.scheduler.lower(), config.scheduler.lower())
        factories = _available_schedulers()
        if alias not in factories:
            raise ValueError(f"Unsupported scheduler: {config.scheduler}")
        scheduler_cls = factories[alias]
        pipeline.scheduler = scheduler_cls.from_config(pipeline.scheduler.config)

    if config.lora_weights is not None:
        pipeline.load_lora_weights(str(config.lora_weights))
        with contextlib.suppress(Exception):  # pragma: no cover - optional optimisation
            if hasattr(pipeline, "fuse_lora"):
                pipeline.fuse_lora()

    device = _select_device(config.device)
    pipeline.to(device)

    return pipeline


def _build_generators(config: GenerationConfig, count: int, device: str) -> List[Any] | None:
    if torch is None or config.seed is None:
        return None
    generators: List[Any] = []
    for index in range(count):
        generator = torch.Generator(device=device)
        generator.manual_seed(config.seed + index)
        generators.append(generator)
    return generators


def generate_images(
    prompts: Sequence[str],
    config: GenerationConfig,
    *,
    pipeline: Any | None = None,
) -> List[Path]:
    """Generate images for ``prompts`` storing them on disk."""

    if not prompts:
        raise ValueError("At least one prompt must be provided for generation.")

    config.prepare()

    active_pipeline = pipeline or build_generation_pipeline(config)
    device = _select_device(config.device)
    generators = _build_generators(config, len(prompts), device)

    saved_paths: List[Path] = []
    for index, prompt in enumerate(prompts):
        prompt_generators = None
        if generators is not None:
            prompt_generators = [generators[index]] * config.images_per_prompt

        run_kwargs: MutableMapping[str, Any] = {
            "prompt": prompt,
            "num_inference_steps": config.num_inference_steps,
            "guidance_scale": config.guidance_scale,
            "height": config.height,
            "width": config.width,
            "num_images_per_prompt": config.images_per_prompt,
        }
        if config.negative_prompt:
            run_kwargs["negative_prompt"] = config.negative_prompt
        if prompt_generators is not None:
            run_kwargs["generator"] = prompt_generators
        if config.extra_kwargs:
            run_kwargs.update(config.extra_kwargs)

        result = active_pipeline(**run_kwargs)
        images: Iterable[Any]
        if isinstance(result, Mapping) and "images" in result:
            images = result["images"]
        else:
            images = getattr(result, "images", result)

        for image_index, image in enumerate(images):
            filename = f"{index:03d}_{image_index:02d}_{_sanitize_prompt(prompt)}.png"
            path = config.output_dir / filename
            image.save(path)
            saved_paths.append(path)
            logger.info("Saved image for prompt '%s' to %s", prompt, path)

    return saved_paths

