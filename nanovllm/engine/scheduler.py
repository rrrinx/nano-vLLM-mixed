from collections import deque
import logging
from .sequence import SequenceStatus
from .batch import BatchPlan, BatchEntry


logger = logging.getLogger(__name__)

def postprocess_one(seq, token_id, is_prefill=None):
    target = seq.num_tokens
    seq.num_cached_tokens += seq.num_scheduled_tokens
    seq.num_scheduled_tokens = 0

    if seq.num_cached_tokens < target:
        return
    if type(token_id) is not int:
        raise ValueError("a complete token interval needs a sampled token")

    seq.append_token(token_id)
    hit_eos = seq.eos_token_id is not None and token_id == seq.eos_token_id
    hit_budget = seq.num_completion_tokens >= seq.max_tokens

    if hit_eos or hit_budget:
        seq.status = SequenceStatus.FINISHED
    else:
        seq.status = SequenceStatus.RUNNING


class Scheduler:
    def __init__(self, cache_manager, max_num_seqs, max_num_batched_tokens, max_model_len, scheduling_policy="separate"):
        self.cache_manager = cache_manager

        limits = (max_num_seqs, max_num_batched_tokens, max_model_len,
                  cache_manager.block_size, cache_manager.num_blocks)
        if any(type(x) is not int or x <= 0 for x in limits):
            raise ValueError("scheduler limits must be positive integers")
        if cache_manager.block_manager.tables or cache_manager.layer_lengths:
            raise ValueError("use an empty cache manager for this engine")
        if scheduling_policy not in ("separate", "interleaved", "mixed"):
            raise ValueError("unknown scheduling policy")

        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.scheduling_policy = scheduling_policy
        self.next_phase = "prefill"

        self.waiting = deque()
        self.running = deque()
        self.request = {}
        self.seen_ids = set()
        self.inflight = None


    def _valid_new(self, seq):
        assert self.inflight is None and seq.seq_id not in self.seen_ids

        assert seq.num_prompt_tokens == seq.num_tokens
        assert seq.num_cached_tokens == 0 and seq.num_scheduled_tokens == 0
        block_size, num_blocks = self.cache_manager.block_size, self.cache_manager.num_blocks
        if seq.max_tokens > 0:
            T_max = seq.max_tokens + seq.num_prompt_tokens - 1
            assert seq.status == SequenceStatus.WAITING
            assert T_max <= min(self.max_model_len, block_size * num_blocks)


    def add(self, seq):
        self._valid_new(seq)
        self.seen_ids.add(seq.seq_id)

        if seq.max_tokens == 0:
            seq.status = SequenceStatus.FINISHED
            return

        self.cache_manager.register([seq])
        self.request[seq.seq_id] = seq
        self.waiting.append(seq)

    def _schedule_prefill(self, token_budget, seq_budget):
        selected = []
        used = 0

        while self.waiting and len(selected) < seq_budget:
            remaining = token_budget - used

            if remaining <= 0:
                break

            seq = self.waiting[0]

            fresh = not self.cache_manager.block_manager.table(seq.seq_id)

            if fresh:
                flag, hits = self.cache_manager.can_allocate_prefill(seq)
                if not flag:
                    break
                c = len(hits) * self.cache_manager.block_size
            else:
                c = seq.num_cached_tokens

            missing = seq.num_tokens - c

            assert missing > 0 and seq.num_scheduled_tokens == 0

            # Only the first request of the prefill sub-batch may be chunked.
            if selected and missing > remaining:
                break

            if fresh:
                attached = self.cache_manager.allocate_prefill(seq, hits)
                assert attached == c
                seq.num_cached_tokens = attached

            q = min(missing, remaining) 
            seq.num_scheduled_tokens = q
            selected.append(seq)
            used += q

            if c + q == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                self.waiting.popleft()
            else:
                assert seq.status == SequenceStatus.WAITING
                break
        return selected

    def _schedule_decode(self, token_budget, seq_budget):
        selected = []
        limit = min(token_budget, seq_budget)
        while self.running and len(selected) < limit:
            seq = self.running.popleft()
            while not self.cache_manager.can_allocate_decode(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                self.cache_manager.allocate_decode(seq)
                seq.num_scheduled_tokens = 1
                selected.append(seq)

        self.running.extendleft(reversed(selected))
        return selected

    def preempt(self, seq):
        self.cache_manager.release(seq.seq_id)
        seq.num_cached_tokens = 0
        seq.num_scheduled_tokens = 0
        seq.status = SequenceStatus.WAITING
        self.cache_manager.register([seq])
        self.waiting.appendleft(seq)


    def is_finished(self):
        return not self.waiting and not self.running and self.inflight is None

    def schedule(self):
        if self.inflight is not None:
            raise ValueError("previous batch has not been committed")
        if self.is_finished():
            return None

        T, S = self.max_num_batched_tokens, self.max_num_seqs
        tagged = []

        def select(phase, tokens=T, slots=S):
            choose = self._schedule_prefill if phase == "prefill" else self._schedule_decode
            return [(seq, phase) for seq in choose(tokens, slots)]

        if self.scheduling_policy == "mixed":
            tagged = select("decode")
            d = len(tagged)
            tagged += select("prefill", T - d, S - d)
        else:
            first = "prefill" if self.scheduling_policy == "separate" else self.next_phase
            second = "decode" if first == "prefill" else "prefill"
            tagged = select(first)
            if not tagged:
                tagged = select(second)


        if not tagged:
            tagged = select("prefill")
        if not tagged:
            tagged = [(seq, "prefill") for seq in self._resume_reserved_prefill(T, S)]
        if not tagged:
            raise RuntimeError("unfinished request made no progress")

        plan = BatchPlan(tuple(BatchEntry(s, phase, s.num_cached_tokens,
                                        s.num_scheduled_tokens, s.num_tokens)
                            for s, phase in tagged))




        plan.validate_live()
        assert len(plan.entries) <= S and 0 < plan.query_tokens <= T
        self.inflight = plan
        # logger.info("scheduled tokens: %s", [seq.num_scheduled_tokens for seq in seqs])
        # logger.info("Scheduled requests: %s", [seq.seq_id for seq in seqs])
        # logger.info("free_blocks: %s", len(self.cache_manager.block_manager.free_block_id))
        # logger.info(" ")

        return plan

    def _resume_reserved_prefill(self, token_budget, seq_budget):
        for seq in self.waiting:
            if (self.cache_manager.block_manager.table(seq.seq_id)
                    and seq.num_cached_tokens < seq.num_tokens):
                candidate = seq
                break
        else:
            return []
        self.waiting.remove(candidate)
        self.waiting.appendleft(candidate)
        return self._schedule_prefill(token_budget, seq_budget)

    postprocess_one = staticmethod(postprocess_one)


    def postprocess(self, plan, token_ids):
        if plan is not self.inflight:
            raise ValueError("result does not belong to the inflight plan")
        plan.validate_live()
        if len(token_ids) != len(plan.entries):
            raise ValueError("one result per batch entry is required")
        for entry, token in zip(plan.entries, token_ids):
            seq = entry.seq
            if self.request.get(seq.seq_id) is not seq:
                raise ValueError("request identity changed")
            if entry.should_sample:
                if type(token) is not int:
                    raise ValueError("sampled result must be an integer")
            elif token is not None:
                raise ValueError("partial prefill must return None")
            expected = entry.cache_len + entry.query_len
            if self.cache_manager.lengths(seq.seq_id) != expected:
                raise RuntimeError("not every layer completed this request")

        for entry in plan.entries:
            self.cache_manager.publish_seq(entry.seq)

        for entry, token in zip(plan.entries, token_ids):
            seq = entry.seq
            postprocess_one(seq, token)
            if seq.status == SequenceStatus.FINISHED:
                self.running.remove(seq)
                self.request.pop(seq.seq_id)
                self.cache_manager.release(seq.seq_id)
        if self.scheduling_policy == "interleaved":
            self.next_phase = "decode" if plan.kind == "prefill" else "prefill"
        self.inflight = None

    def abort(self):
        for sid, seq in list(self.request.items()):
            self.cache_manager.release(sid)
            seq.num_cached_tokens = 0
            seq.num_scheduled_tokens = 0
        self.request.clear()
        self.waiting.clear()
        self.running.clear()
        self.inflight = None
        self.next_phase = "prefill"
