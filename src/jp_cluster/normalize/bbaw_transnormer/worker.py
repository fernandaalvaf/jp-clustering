"""
GPU worker for Transnormer inference.

Each worker owns one model replica on a specific GPU.
It pulls sentence batches from a shared work queue, runs inference,
and pushes results to a result queue.  Workers are fully async —
fast workers keep pulling work without waiting for slow ones.
"""

import logging
import time
from multiprocessing import Queue
from multiprocessing.synchronize import Event as EventType

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

logger = logging.getLogger(__name__)

# Sentinel value that tells a worker to shut down
_POISON = None


def worker_loop(
    worker_id: int,
    gpu_id: int,
    model_name: str,
    work_queue: Queue,
    result_queue: Queue,
    ready_event: EventType,
    max_new_tokens: int = 512,
    num_beams: int = 1,
    use_compile: bool = True,
    use_fp16: bool = True,
    length_multiplier: float = 1.3,
) -> None:
    """
    Long-running worker process.

    1. Loads & compiles the model on ``cuda:<gpu_id>``.
    2. Signals *ready_event* so the orchestrator knows loading is done.
    3. Loops: pull a work item from *work_queue*, run inference,
       push ``(batch_id, list[str])`` to *result_queue*.
    4. Exits when it receives the ``_POISON`` sentinel.

    Work items on the queue are tuples::

        (batch_id: int, texts: list[str])

    ``batch_id`` lets the orchestrator map results back to the
    original document positions.
    """
    # Spawned processes don't inherit the parent's logging config
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (use_fp16 and device.startswith("cuda")) else torch.float32

    # ---- Load model & tokenizer ----
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=dtype).to(device)
    model.eval()

    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.num_beams = num_beams
    # max_new_tokens is set per-batch from input length (see loop below)

    # ---- Optional torch.compile ----
    if use_compile and hasattr(torch, "compile"):
        try:
            logger.info(f"[worker {worker_id}] starting torch.compile on {device} …")
            t_compile = time.perf_counter()
            model = torch.compile(model, mode="default")
            logger.info(
                f"[worker {worker_id}] torch.compile ok on {device} "
                f"in {time.perf_counter() - t_compile:.1f}s"
            )
        except Exception as exc:
            logger.warning(f"[worker {worker_id}] torch.compile failed: {exc}")

    load_time = time.perf_counter() - t0
    logger.info(f"[worker {worker_id}] model loaded on {device} in {load_time:.1f}s")

    # Signal that this worker is ready
    ready_event.set()

    # ---- Inference loop ----
    batches_done = 0
    with torch.inference_mode():
        while True:
            item = work_queue.get()
            if item is _POISON:
                break

            batch_id, idxs, texts = item

            if batches_done == 0:
                logger.info(
                    f"[worker {worker_id}] processing first batch "
                    f"(compile overhead may apply) …"
                )

            t_batch = time.perf_counter()

            # Tokenize WITHOUT silent truncation: we'd rather see an OOM
            # than drop bytes from the input. Upstream chunking in
            # _split_sentences should keep us safely below the model's
            # input window.
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(device)

            longest_input = int(inputs["input_ids"].shape[1])
            model_max = getattr(tokenizer, "model_max_length", None) or 1024
            if longest_input > model_max:
                logger.warning(
                    f"[worker {worker_id}] batch {batch_id}: longest input "
                    f"{longest_input} > model_max_length {model_max}; "
                    f"output may be incomplete"
                )

            # Per-batch max_new_tokens scales with input length so the
            # decoder always has enough budget to reproduce the text.
            batch_max_new = min(
                max_new_tokens,
                max(32, int(longest_input * length_multiplier) + 8),
            )
            gen_cfg.max_new_tokens = batch_max_new

            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=gen_cfg,
            )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Sanity check: warn if any output is suspiciously short
            # relative to its input. This catches early-EOS failures
            # like the model bailing on weird prefixes.
            for src, out in zip(texts, decoded):
                src_len = len(src.encode("utf-8"))
                out_len = len(out.encode("utf-8"))
                if src_len >= 40 and out_len < src_len * 0.5:
                    logger.warning(
                        f"[worker {worker_id}] batch {batch_id}: "
                        f"short output {out_len}B for input {src_len}B "
                        f"(input starts: {src[:60]!r})"
                    )

            result_queue.put((batch_id, idxs, decoded))

            if batches_done == 0:
                logger.info(
                    f"[worker {worker_id}] first batch done in "
                    f"{time.perf_counter() - t_batch:.1f}s"
                )

            batches_done += 1

    logger.info(f"[worker {worker_id}] exiting after {batches_done} batches")
