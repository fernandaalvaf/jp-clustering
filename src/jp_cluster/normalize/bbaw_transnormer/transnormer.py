"""
Optimized Transnormer orchestrator.

Supports:
  - Multi-GPU: spreads workers across all available GPUs.
  - Multiple models per GPU: N replicas share one GPU.
  - Async work queue: fast workers keep pulling; no one waits
    for the slowest.
  - torch.compile: JIT compilation for faster inference.
  - Flat sentence batching: all documents are split into
    sentences, batched globally, and reassembled after inference.
  - FP16 inference by default.

Usage (CLI)::

    python transnormer.py raw/ --output results/  \
        --models-per-gpu 2 --batch-size 32

Usage (Library)::

    from transnormer import normalize_files
    results = normalize_files(
        input_dir="raw/",
        models_per_gpu=2,
        batch_size=32,
    )
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing as mp
import time
from datetime import datetime
from multiprocessing import Process, Queue
from pathlib import Path

import torch

from jp_cluster.normalize.bbaw_transnormer.worker import worker_loop
from jp_cluster.utils.text_splitting import split_sentences

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "ybracke/transnormer-19c-beta-v01"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_NEW_TOKENS = 512  # absolute cap; per-batch is computed dynamically
DEFAULT_NUM_BEAMS = 4 # greedy: byte-level beam search is brutally expensive
DEFAULT_LENGTH_MULTIPLIER = 1.5  # output bytes ≈ input bytes * this


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _read_files(input_path: Path) -> dict[str, str]:
    """
    Read input documents.

    Accepts either:
      * a directory of ``.txt`` files (key = file stem), or
      * a single ``.json`` file with a list of records carrying
        ``document_id`` (or ``id``/``xml_path`` fallback) and
        ``raw_text`` (or ``text``).
    """
    if input_path.is_file() and input_path.suffix.lower() == ".json":
        return _read_json(input_path)
    if input_path.is_dir():
        texts: dict[str, str] = {}
        for p in sorted(input_path.glob("*.txt")):
            try:
                texts[p.stem] = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                texts[p.stem] = p.read_text(encoding="cp1252", errors="replace")
        return texts
    raise ValueError(
        f"Input must be a directory of .txt files or a .json file: {input_path}"
    )


def _read_json(json_path: Path) -> dict[str, str]:
    """Load documents from a JSON array of ``{document_id, raw_text}`` records."""
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{json_path}: expected a JSON array of records")

    texts: dict[str, str] = {}
    for i, rec in enumerate(raw):
        if not isinstance(rec, dict):
            logger.warning("%s[%d]: not an object, skipping", json_path.name, i)
            continue
        text = rec.get("raw_text") or rec.get("text")
        if not text:
            logger.warning("%s[%d]: no raw_text/text field, skipping", json_path.name, i)
            continue
        key = rec.get("document_id") or rec.get("id")
        if not key:
            xml_path = rec.get("xml_path")
            key = Path(xml_path.replace("\\", "/")).stem if xml_path else f"doc_{i:05d}"
        if key in texts:
            logger.warning("Duplicate key %r in %s; later record wins", key, json_path.name)
        texts[key] = text
    return texts


def _flatten_to_sentences(
    texts: dict[str, str],
) -> tuple[list[str], list[tuple[str, int]]]:
    """
    Flatten all documents into a flat sentence list.

    Returns
    -------
    sentences : list[str]
        Every sentence across all documents.
    mapping : list[tuple[str, int]]
        ``(file_key, sentence_index_within_file)`` for each sentence
        so we can reassemble later.
    """
    sentences: list[str] = []
    mapping: list[tuple[str, int]] = []
    for key, text in texts.items():
        sents = split_sentences(text)
        for i, s in enumerate(sents):
            sentences.append(s)
            mapping.append((key, i))
    return sentences, mapping


def _make_batches(
    sentences: list[str],
    batch_size: int,
    *,
    length_bucketed: bool = True,
) -> list[tuple[int, list[int], list[str]]]:
    """
    Group sentences into numbered batches.

    When ``length_bucketed`` is True (default), sentences are sorted by
    UTF-8 byte length before slicing so each batch contains
    similarly-sized inputs. This drastically reduces padding waste on
    byte-level models.

    Returns a list of ``(batch_id, original_indices, texts)`` ready for
    the work queue. ``original_indices`` lets the orchestrator place
    decoded outputs back into their original positions.
    """
    n = len(sentences)
    order = list(range(n))
    if length_bucketed:
        order.sort(key=lambda i: len(sentences[i].encode("utf-8")))

    batches: list[tuple[int, list[int], list[str]]] = []
    for bid, start in enumerate(range(0, n, batch_size)):
        idxs = order[start : start + batch_size]
        texts = [sentences[i] for i in idxs]
        batches.append((bid, idxs, texts))
    return batches


def _reassemble(
    results: dict[int, tuple[list[int], list[str]]],
    total_sentences: int,
    mapping: list[tuple[str, int]],
    original_texts: dict[str, str],
) -> dict[str, dict]:
    """
    Reassemble flat normalized sentences back into per-document output.
    """
    # Place each decoded sentence back at its original index
    normalized: list[str] = [""] * total_sentences
    for _batch_id, (idxs, decoded_list) in results.items():
        for orig_idx, text in zip(idxs, decoded_list):
            normalized[orig_idx] = text

    # Group by file key
    doc_sentences: dict[str, list[str]] = {}
    for idx, (key, _sent_idx) in enumerate(mapping):
        doc_sentences.setdefault(key, []).append(normalized[idx])

    # Build output dict
    output: dict[str, dict] = {}
    for key in original_texts:
        norm_text = " ".join(doc_sentences.get(key, []))
        output[key] = {
            "original_text": original_texts[key],
            "normalized_text": norm_text,
        }
    return output


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def normalize_files(
    input_dir: str | Path,
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    num_beams: int = DEFAULT_NUM_BEAMS,
    num_gpus: int | None = None,
    models_per_gpu: int = 4,
    use_compile: bool = True,
    use_fp16: bool = False,
    length_bucketed: bool = True,
    length_multiplier: float = DEFAULT_LENGTH_MULTIPLIER,
    benchmark: bool = False,
) -> dict[str, dict] | tuple[dict[str, dict], dict]:
    """
    Normalize all .txt files in *input_dir*.

    Parameters
    ----------
    input_dir
        Folder containing ``.txt`` files.
    model_name
        HuggingFace model identifier.
    batch_size
        Sentences per inference batch.
    max_new_tokens / num_beams
        Generation parameters.
    num_gpus
        GPUs to use (``None`` = auto-detect all).
    models_per_gpu
        Model replicas per GPU.  Increase until GPU memory is full
        for maximum throughput.
    use_compile
        Apply ``torch.compile`` (PyTorch ≥ 2.0).
    use_fp16
        Use float16 on CUDA.
    benchmark
        When ``True``, return ``(results, stats)`` where *stats* is
        a dict of timing metrics suitable for CSV export.

    Returns
    -------
    dict  mapping ``file_stem → {original_text, normalized_text}``
        When *benchmark* is ``True``, returns
        ``(results_dict, stats_dict)`` instead.
    """
    input_dir = Path(input_dir)
    t_total_start = time.perf_counter()

    # ---- Read & flatten ----
    t_read_start = time.perf_counter()
    texts = _read_files(input_dir)
    t_read = time.perf_counter() - t_read_start
    if not texts:
        logger.warning("No input documents found in %s", input_dir)
        return ({}, {}) if benchmark else {}

    sentences, mapping = _flatten_to_sentences(texts)
    batches = _make_batches(sentences, batch_size, length_bucketed=length_bucketed)
    total_sentences = len(sentences)
    total_batches = len(batches)

    logger.info(
        f"{len(texts)} files → {total_sentences} sentences → {total_batches} batches"
    )

    # ---- Detect GPUs ----
    if torch.cuda.is_available():
        available = torch.cuda.device_count()
        if num_gpus is None:
            num_gpus = available
        else:
            num_gpus = min(num_gpus, available)
    else:
        num_gpus = 0  # CPU fallback below

    total_workers = max(num_gpus * models_per_gpu, 1)  # at least 1 (CPU)
    logger.info(
        f"Using {num_gpus} GPU(s) × {models_per_gpu} model(s) = {total_workers} worker(s)"
    )

    # ---- Spawn workers ----
    # CUDA requires 'spawn'
    t_spawn_start = time.perf_counter()
    ctx = mp.get_context("spawn")
    work_queue: Queue = ctx.Queue()
    result_queue: Queue = ctx.Queue()

    workers: list[Process] = []
    ready_events: list = []

    for w in range(total_workers):
        gpu_id = w // models_per_gpu if num_gpus > 0 else 0
        evt = ctx.Event()
        ready_events.append(evt)

        p = ctx.Process(
            target=worker_loop,
            kwargs=dict(
                worker_id=w,
                gpu_id=gpu_id,
                model_name=model_name,
                work_queue=work_queue,
                result_queue=result_queue,
                ready_event=evt,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                use_compile=use_compile,
                use_fp16=use_fp16,
                length_multiplier=length_multiplier,
            ),
            daemon=True,
        )
        p.start()
        workers.append(p)

    # Wait for all workers to finish loading
    logger.info("Waiting for all workers to load models …")
    for evt in ready_events:
        evt.wait()
    logger.info("All workers ready — feeding work queue")
    t_worker_load = time.perf_counter() - t_spawn_start

    # ---- Feed the work queue ----
    t0 = time.perf_counter()
    for batch in batches:
        work_queue.put(batch)

    # Send one poison pill per worker
    for _ in workers:
        work_queue.put(None)

    # ---- Collect results (async — first done, first collected) ----
    collected: dict[int, tuple[list[int], list[str]]] = {}
    for _ in range(total_batches):
        batch_id, idxs, decoded = result_queue.get()
        collected[batch_id] = (idxs, decoded)

    elapsed = time.perf_counter() - t0
    logger.info(
        f"Inference done: {total_batches} batches in {elapsed:.1f}s "
        f"({total_sentences / elapsed:.0f} sent/s)"
    )

    # ---- Wait for workers to exit ----
    for p in workers:
        p.join(timeout=30)

    # ---- Reassemble ----
    t_reassemble_start = time.perf_counter()
    output = _reassemble(collected, total_sentences, mapping, texts)
    t_reassemble = time.perf_counter() - t_reassemble_start
    t_total = time.perf_counter() - t_total_start

    if benchmark:
        stats = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": model_name,
            "num_files": len(texts),
            "num_sentences": total_sentences,
            "num_batches": total_batches,
            "batch_size": batch_size,
            "num_gpus": num_gpus,
            "num_workers": total_workers,
            "use_fp16": use_fp16,
            "use_compile": use_compile,
            "read_time_s": round(t_read, 3),
            "worker_load_time_s": round(t_worker_load, 3),
            "inference_time_s": round(elapsed, 3),
            "reassembly_time_s": round(t_reassemble, 3),
            "total_time_s": round(t_total, 3),
            "sentences_per_sec": (
                round(total_sentences / elapsed, 1) if elapsed > 0 else 0
            ),
        }
        return output, stats
    return output


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------


def _write_benchmark_csv(csv_path: Path, stats: dict) -> None:
    """Append one benchmark row to *csv_path* (creates file + header if needed)."""
    fieldnames = [
        "timestamp",
        "model",
        "num_files",
        "num_sentences",
        "num_batches",
        "batch_size",
        "num_gpus",
        "num_workers",
        "use_fp16",
        "use_compile",
        "read_time_s",
        "worker_load_time_s",
        "inference_time_s",
        "reassembly_time_s",
        "total_time_s",
        "sentences_per_sec",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(stats)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Normalize historical German texts with Transnormer"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory with .txt input files, or a .json file with "
        "[{document_id, raw_text}, ...] records",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: sibling 'normalized/' next to input)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--num-beams", type=int, default=DEFAULT_NUM_BEAMS)
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="GPUs to use (default: all available)",
    )
    parser.add_argument(
        "--models-per-gpu",
        type=int,
        default=1,
        help="Model replicas per GPU (increase to fill GPU memory)",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable float16 inference",
    )
    parser.add_argument(
        "--no-length-bucket",
        action="store_true",
        help="Disable length-bucketed batching (sort sentences by length "
        "before batching to minimize padding waste)",
    )
    parser.add_argument(
        "--length-multiplier",
        type=float,
        default=DEFAULT_LENGTH_MULTIPLIER,
        help="Per-batch max_new_tokens = max_input_len * this (capped by "
        "--max-new-tokens). Output bytes ≈ input bytes for normalization.",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=None,
        help="Path to benchmark CSV; appends one row per run",
    )
    args = parser.parse_args()

    # Default output: sibling 'normalized/' next to the input (works for
    # both a directory of .txt files and a single .json file).
    if args.output is not None:
        output_dir = args.output
    elif args.input_dir.is_file():
        output_dir = args.input_dir.parent / "normalized"
    else:
        output_dir = args.input_dir.parent / "normalized"
    output_dir.mkdir(parents=True, exist_ok=True)

    do_benchmark = args.benchmark is not None
    result = normalize_files(
        input_dir=args.input_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        num_gpus=args.num_gpus,
        models_per_gpu=args.models_per_gpu,
        use_compile=not args.no_compile,
        use_fp16=not args.no_fp16,
        length_bucketed=not args.no_length_bucket,
        length_multiplier=args.length_multiplier,
        benchmark=do_benchmark,
    )

    if do_benchmark:
        assert isinstance(result, tuple)
        results, stats = result
        _write_benchmark_csv(args.benchmark, stats)
        logger.info(f"Benchmark row appended to {args.benchmark}")
    else:
        assert isinstance(result, dict)
        results = result

    # Write outputs
    for key, data in results.items():
        out_file = output_dir / f"{key}.normalized.txt"
        out_file.write_text(data["normalized_text"], encoding="utf-8")

    # Write timing summary
    summary_file = output_dir / "summary.json"
    summary = {
        k: {
            "original_length": len(v["original_text"]),
            "normalized_length": len(v["normalized_text"]),
        }
        for k, v in results.items()
    }
    summary_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
