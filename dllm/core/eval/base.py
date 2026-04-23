"""
Generic eval harness base: accelerator, rank/world_size, model/tokenizer loading,
device, apply_chat_template, tokenizer_name, unified generate_until scaffolding.
Pipeline-agnostic; no MDLM/Dream specifics.

Run: Not runnable directly; use pipeline eval entrypoints (e.g. dllm.pipelines.llada.eval).
"""

import dataclasses
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass

import accelerate
import torch
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from tqdm import tqdm

import dllm
from dllm.core.samplers import BaseSampler, BaseSamplerConfig
from dllm.utils.configs import ModelArguments


@dataclass
class BaseEvalConfig:
    """Minimal config for base eval: device and batch_size."""

    pretrained: str = ""
    device: str = "cuda"
    batch_size: int = 1

    def get_model_config(self, pretrained: str):
        """Optional: return custom model config for loading. Default None (use checkpoint config)."""
        return None


class BaseEvalHarness(LM):
    """
    Pipeline-agnostic eval base: accelerator, rank/world_size, model and tokenizer
    loading, device placement, apply_chat_template, tokenizer_name.
    Subclasses implement loglikelihood (and optionally loglikelihood_rolling);
    generate_until is implemented here and uses sampler + sampler_config.
    """

    @staticmethod
    def _build_config(config_cls, source, kwargs):
        """Build a dataclass *config_cls* by copying fields from *source*, with *kwargs* overrides."""
        init = {}
        for f in dataclasses.fields(config_cls):
            if f.name in kwargs:
                init[f.name] = kwargs[f.name]
            elif hasattr(source, f.name):
                init[f.name] = getattr(source, f.name)
        return config_cls(**init)

    def __init__(
        self,
        eval_config: BaseEvalConfig | None = None,
        model_args: ModelArguments | None = None,
        sampler_config: BaseSamplerConfig | None = None,
        sampler_cls: type[BaseSampler] | None = None,
        **kwargs,
    ):
        super().__init__()
        eval_config = eval_config or BaseEvalConfig()
        # Ensure model path is in kwargs and we have a safe default for ModelArguments(__post_init__).
        model_args = model_args or ModelArguments(
            model_name_or_path=kwargs.get("pretrained")
        )
        device = kwargs.get("device", eval_config.device)

        # ── Distributed ──────────────────────────────────────────
        accelerator = accelerate.Accelerator()
        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()
        else:
            self._rank = 0
            self._world_size = 1

        # ── Model + tokenizer + sampler ──────────────────────────
        if "pretrained" in kwargs:
            kwargs.setdefault("model_name_or_path", kwargs["pretrained"])
        self.model_args = self._build_config(ModelArguments, model_args, kwargs)
        self.model = dllm.utils.get_model(
            self.model_args,
            config=eval_config.get_model_config(self.model_args.model_name_or_path),
        )
        self.model.eval()
        self.tokenizer = dllm.utils.get_tokenizer(self.model_args)
        if sampler_config is not None:
            self.sampler_config = self._build_config(
                type(sampler_config), sampler_config, kwargs
            )
        if sampler_cls is not None:
            self.sampler = sampler_cls(model=self.model, tokenizer=self.tokenizer)

        # ── Device placement ─────────────────────────────────────
        if accelerator.num_processes > 1:
            self.model = accelerator.prepare(self.model)
            self.device = accelerator.device
            self.accelerator = accelerator
        else:
            self.model = self.model.to(device)
            self.device = torch.device(device)
            self.accelerator = None

        self.batch_size = int(kwargs.get("batch_size", eval_config.batch_size))
        self.num_workers = int(kwargs.get("num_workers", 1))
        self.output_dir = kwargs.get("output_dir", None)

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def apply_chat_template(
        self,
        chat_history: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Format chat history for input to the LM."""
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

    # ── Unified generate_until scaffolding ────────────────────────────

    def _save_generation(self, request: Instance, answer: str) -> None:
        """Append a completed generation to the incremental JSONL log."""
        if self.output_dir is None:
            return
        path = os.path.join(self.output_dir, "generations.jsonl")
        line = json.dumps(
            {
                "doc_id": request.doc_id,
                "task_name": request.task_name,
                "context": request.args[0],
                "generated": answer,
            },
            ensure_ascii=False,
        )
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _process_request(self, request: Instance, stream) -> str:
        """Process a single generation request, optionally on a dedicated CUDA stream."""
        ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with torch.no_grad(), ctx:
            context, gen_kwargs = request.args
            prompt = torch.tensor(
                self.tokenizer(context)["input_ids"],
                device=self.device,
                dtype=torch.long,
            )
            generated_ids = self.sampler.sample(
                inputs=[prompt],
                config=self.sampler_config,
                return_dict=False,
            )
            answer = dllm.utils.sample_trim(
                self.tokenizer,
                generated_ids.tolist(),
                [prompt.tolist()],
            )[0]
            for stop_seq in gen_kwargs["until"]:
                if stop_seq in answer:
                    answer = answer.split(stop_seq)[0]
            return answer

    @torch.no_grad()
    def generate_until(self, requests: list[Instance]) -> list[str]:
        if self.num_workers > 1:
            return self._generate_until_parallel(requests)

        out: list[str] = []

        for batch_start in tqdm(
            range(0, len(requests), self.batch_size), desc="Generating..."
        ):
            batch = requests[batch_start : batch_start + self.batch_size]
            contexts, gen_kwargs_list = zip(*[inst.args for inst in batch])

            prompts = [
                torch.tensor(
                    self.tokenizer(ctx)["input_ids"],
                    device=self.device,
                    dtype=torch.long,
                )
                for ctx in contexts
            ]

            generated_ids = self.sampler.sample(
                inputs=prompts,
                config=self.sampler_config,
                return_dict=False,
            )
            generated_answers = dllm.utils.sample_trim(
                self.tokenizer,
                generated_ids.tolist(),
                [p.tolist() for p in prompts],
            )

            for inst, answer, gen_kwargs in zip(batch, generated_answers, gen_kwargs_list):
                for stop_seq in gen_kwargs["until"]:
                    if stop_seq in answer:
                        answer = answer.split(stop_seq)[0]
                self._save_generation(inst, answer)
                out.append(answer)

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out

    def _generate_until_parallel(self, requests: list[Instance]) -> list[str]:
        """Process requests concurrently with a thread pool and per-thread CUDA streams."""
        if self.device.type == "cuda":
            streams = [torch.cuda.Stream(device=self.device) for _ in range(self.num_workers)]
        else:
            streams = [None] * self.num_workers

        results: list[str | None] = [None] * len(requests)

        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = {}
            for i, req in enumerate(requests):
                stream = streams[i % self.num_workers]
                future = pool.submit(self._process_request, req, stream)
                futures[future] = i

            for future in tqdm(as_completed(futures), total=len(requests), desc="Generating..."):
                idx = futures[future]
                answer = future.result()
                self._save_generation(requests[idx], answer)
                results[idx] = answer

        return results

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
