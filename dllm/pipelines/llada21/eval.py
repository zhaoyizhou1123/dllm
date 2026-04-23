"""
LLaDA 2.1 evaluation harness for lm-evaluation-harness.

Models:
    llada21                 — LLaDA21Sampler (iterative editing, batch_size=1)
    llada21_block           — LLaDA21BlockSampler (fixed-schedule block diffusion)
    llada21_gibbs_block     — LLaDA21GibbsBlockSampler (fixed-schedule + Gibbs correction)
    llada21_confidence_block — LLaDA21ConfidenceBlockSampler (confidence-based + Gibbs correction)
"""

from dataclasses import dataclass

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from dllm.core.eval import BaseEvalConfig, BaseEvalHarness
from dllm.pipelines.llada21 import (
    LLaDA21BlockSampler,
    LLaDA21BlockSamplerConfig,
    LLaDA21ConfidenceBlockSampler,
    LLaDA21ConfidenceBlockSamplerConfig,
    LLaDA21GibbsBlockSampler,
    LLaDA21GibbsBlockSamplerConfig,
    LLaDA21Sampler,
    LLaDA21SamplerConfig,
)


@dataclass
class LLaDA21EvalSamplerConfig(LLaDA21SamplerConfig):
    """Default sampler config for LLaDA 2.1 eval (HF Speed Mode defaults)."""

    max_new_tokens: int = 512
    block_size: int = 32
    temperature: float = 0.0
    threshold: float = 0.5
    editing_threshold: float = 0.0
    max_post_steps: int = 16
    num_to_transfer: int = 1
    eos_early_stop: bool = True


@dataclass
class LLaDA21EvalConfig(BaseEvalConfig):
    """LLaDA 2.1 eval config. Batch size forced to 1."""

    batch_size: int = 1


@register_model("llada21")
class LLaDA21EvalHarness(BaseEvalHarness):
    def __init__(
        self,
        eval_config: LLaDA21EvalConfig | None = None,
        sampler_config: LLaDA21SamplerConfig | None = None,
        sampler_cls: type[LLaDA21Sampler] = LLaDA21Sampler,
        **kwargs,
    ):
        eval_config = eval_config or LLaDA21EvalConfig()
        sampler_config = sampler_config or LLaDA21EvalSamplerConfig()

        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )


@dataclass
class LLaDA21BlockEvalSamplerConfig(LLaDA21BlockSamplerConfig):
    """Default sampler config for LLaDA 2.1 standard block diffusion eval."""

    max_new_tokens: int = 512
    block_size: int = 32
    steps_per_block: int = 32
    temperature: float = 0.0
    eos_early_stop: bool = True


@dataclass
class LLaDA21BlockEvalConfig(BaseEvalConfig):
    """LLaDA 2.1 block eval config. Supports batch_size > 1."""

    batch_size: int = 4


@register_model("llada21_block")
class LLaDA21BlockEvalHarness(BaseEvalHarness):
    def __init__(
        self,
        eval_config: LLaDA21BlockEvalConfig | None = None,
        sampler_config: LLaDA21BlockSamplerConfig | None = None,
        sampler_cls: type[LLaDA21BlockSampler] = LLaDA21BlockSampler,
        **kwargs,
    ):
        eval_config = eval_config or LLaDA21BlockEvalConfig()
        sampler_config = sampler_config or LLaDA21BlockEvalSamplerConfig()

        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )


@dataclass
class LLaDA21GibbsBlockEvalSamplerConfig(LLaDA21GibbsBlockSamplerConfig):
    max_new_tokens: int = 512
    block_size: int = 32
    unmasking_num: int = 1
    temperature: float = 0.0
    eos_early_stop: bool = True


@dataclass
class LLaDA21GibbsBlockEvalConfig(BaseEvalConfig):
    batch_size: int = 1


@register_model("llada21_gibbs_block")
class LLaDA21GibbsBlockEvalHarness(BaseEvalHarness):
    def __init__(
        self,
        eval_config: LLaDA21GibbsBlockEvalConfig | None = None,
        sampler_config: LLaDA21GibbsBlockSamplerConfig | None = None,
        sampler_cls: type[LLaDA21GibbsBlockSampler] = LLaDA21GibbsBlockSampler,
        **kwargs,
    ):
        eval_config = eval_config or LLaDA21GibbsBlockEvalConfig()
        sampler_config = sampler_config or LLaDA21GibbsBlockEvalSamplerConfig()

        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )


@dataclass
class LLaDA21ConfidenceBlockEvalSamplerConfig(LLaDA21ConfidenceBlockSamplerConfig):
    max_new_tokens: int = 512
    block_size: int = 32
    threshold: float = 0.9
    min_transfer: int = 1
    temperature: float = 0.0
    eos_early_stop: bool = True


@dataclass
class LLaDA21ConfidenceBlockEvalConfig(BaseEvalConfig):
    batch_size: int = 1


@register_model("llada21_confidence_block")
class LLaDA21ConfidenceBlockEvalHarness(BaseEvalHarness):
    def __init__(
        self,
        eval_config: LLaDA21ConfidenceBlockEvalConfig | None = None,
        sampler_config: LLaDA21ConfidenceBlockSamplerConfig | None = None,
        sampler_cls: type[LLaDA21ConfidenceBlockSampler] = LLaDA21ConfidenceBlockSampler,
        **kwargs,
    ):
        eval_config = eval_config or LLaDA21ConfidenceBlockEvalConfig()
        sampler_config = sampler_config or LLaDA21ConfidenceBlockEvalSamplerConfig()

        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )


if __name__ == "__main__":
    cli_evaluate()
