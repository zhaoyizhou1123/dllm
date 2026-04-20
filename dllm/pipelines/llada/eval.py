"""
accelerate launch \
    --num_processes 4 \
    dllm/pipelines/llada/eval.py \
    --tasks gsm8k_cot \
    --model llada \
    --apply_chat_template \
    --num_fewshot 5 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,max_new_tokens=512,steps=512,block_size=512,cfg_scale=0.0"
"""

from dataclasses import dataclass

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from dllm.core.eval import MDLMEvalConfig, MDLMEvalHarness
from dllm.core.samplers import (
    GibbsSampler,
    GibbsSamplerConfig,
    MDLMSampler,
    MDLMSamplerConfig,
)


@dataclass
class LLaDAEvalSamplerConfig(MDLMSamplerConfig):
    """Default sampler config for LLaDA eval."""

    max_new_tokens: int = 1024
    steps: int = 1024
    block_size: int = 1024


@dataclass
class LLaDAEvalConfig(MDLMEvalConfig):
    """LLaDA eval config."""

    # According to LLaDA's opencompass implementation:
    # https://github.com/ML-GSAI/LLaDA/blob/main/opencompass/opencompass/models/dllm.py
    max_length: int = 4096


@register_model("llada")
class LLaDAEvalHarness(MDLMEvalHarness):
    def __init__(
        self,
        eval_config: LLaDAEvalConfig | None = None,
        sampler_config: MDLMSamplerConfig | None = None,
        sampler_cls: type[MDLMSampler] = MDLMSampler,
        **kwargs,
    ):
        eval_config = eval_config or LLaDAEvalConfig()
        sampler_config = sampler_config or LLaDAEvalSamplerConfig()

        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )


@dataclass
class LLaDAGibbsEvalSamplerConfig(GibbsSamplerConfig):
    """Default sampler config for LLaDA eval with Gibbs correction."""

    max_new_tokens: int = 1024
    steps: int = 1024
    block_size: int = 1024


@register_model("llada_gibbs")
class LLaDAGibbsEvalHarness(MDLMEvalHarness):
    """LLaDA eval harness wired to GibbsSampler.

    Select one of the three Gibbs variants via ``--model_args edit_strategy=...``:
    ``gibbs_standard`` | ``gibbs_edit`` | ``gibbs_edit_v2``. Set ``edit_freq=-1``
    to disable correction (reduces to MDLMSampler behavior).
    """

    def __init__(
        self,
        eval_config: LLaDAEvalConfig | None = None,
        sampler_config: GibbsSamplerConfig | None = None,
        sampler_cls: type[GibbsSampler] = GibbsSampler,
        **kwargs,
    ):
        eval_config = eval_config or LLaDAEvalConfig()
        sampler_config = sampler_config or LLaDAGibbsEvalSamplerConfig()

        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )


if __name__ == "__main__":
    cli_evaluate()
