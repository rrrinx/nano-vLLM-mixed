import atexit
import logging

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from ..config import Config
from ..sampling_params import SamplingParams
from .model_runner import ModelRunner
from .scheduler import Scheduler
from .sequence import Sequence, SequenceStatus

class LLMEngine:
    def __init__(self, model, **kwargs):
        self.config = Config(model, **kwargs)
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.log_level))
        if not root_logger.handlers:
            logging.basicConfig(
                level=getattr(logging, self.config.log_level),
                format="%(asctime)s %(levelname)s: %(message)s",
            )
            # format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        self.model_runner = ModelRunner(self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        self.config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(
            cache_manager=self.model_runner.kv_cache_manager,
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            max_model_len=self.config.max_model_len,
            scheduling_policy=self.config.scheduling_policy,
        )
        self.next_seq_id = 0
        self.ready_outputs = []
        self.failed = False
        atexit.register(self.exit)

    def add_request(self, prompt, sampling_params=None, eos_token_id=None):
        if self.failed:
            raise RuntimeError("engine has failed; create a new engine")

        if sampling_params is None:
            sampling_params = SamplingParams()
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if sampling_params.ignore_eos:
            eos_token_id = None
        elif eos_token_id is None:
            eos_token_id = self.config.eos

        seq = Sequence(
            seq_id=self.next_seq_id,
            token_ids=prompt,
            max_tokens=sampling_params.max_tokens,
            eos_token_id=eos_token_id
        )

        self.scheduler.add(seq)
        self.next_seq_id += 1

        if seq.status == SequenceStatus.FINISHED:
            self.ready_outputs.append((seq.seq_id, list(seq.completion_token_ids)))
        return seq.seq_id

    def is_finished(self):
        if self.failed:
            raise RuntimeError("engine has failed")
        return self.scheduler.is_finished() and not self.ready_outputs

    def step(self):
        if self.failed:
            raise RuntimeError("engine has failed; create a new engine")

        outputs = list(self.ready_outputs) # copy

        try:
            plan = self.scheduler.schedule()
            if plan is not None:
                output_ids = self.model_runner.run_batch(plan)
                self.scheduler.postprocess(plan, output_ids)
                outputs.extend((s.seq_id, list(s.completion_token_ids))
                   for s in plan.seqs if s.status == SequenceStatus.FINISHED)
            self.ready_outputs.clear()

            return outputs
        except Exception:
            self.failed = True
            self.scheduler.abort()
            self.ready_outputs.clear()
            raise

    def generate(self, prompts, sampling_params=None, use_tqdm=True, *, eos_token_id=None):
        if not self.is_finished():
            raise RuntimeError("generate requires an idle engine")

        if sampling_params is None:
            sampling_params = SamplingParams()
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        assert len(prompts) == len(sampling_params), "Each prompt needs one SamplingParams"

        order = []
        output = {}
        pbar = tqdm(
            total=len(prompts),
            desc="Generating",
            disable=not (use_tqdm and self.config.enable_tqdm),
        )

        try:
            for prompt, params in zip(prompts, sampling_params):
                order.append(
                    self.add_request(prompt, params, eos_token_id))

            while not self.is_finished():
                for seq_id, output_ids in self.step():
                    output[seq_id] = output_ids
                    pbar.update(1)

            token_ids = [output[seq_id] for seq_id in order]
            return [{"text": self.tokenizer.decode(ids), "token_ids": ids} for ids in token_ids]
        except Exception:
            self.failed = True
            self.scheduler.abort()
            self.ready_outputs.clear()
            raise
        finally:
            pbar.close()

    def exit(self):
        if self.model_runner is not None:
            self.scheduler.abort()
            self.ready_outputs.clear()
            self.model_runner = None
            self.scheduler = None
