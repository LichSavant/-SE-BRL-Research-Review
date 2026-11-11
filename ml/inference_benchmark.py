"""Synthetic model-only timing port; no fitted model, hard SRS limit or selection.

The trusted callback must be prediction on an already model-compatible numeric
matrix. Parsing/features/calibration/attribution/API/rendering stay outside this
boundary. Timing results are batch latency, not per-artifact tail latency.
"""
import math
from time import perf_counter

from ml.evaluation_metrics import _numbers, _quantile
from ml.experiment_inputs import InputContractError


def benchmark_synthetic_inference(predict, rows, *, warmup, repetitions, batch_size,
                                  thread_reference, serialized_model_bytes=None,
                                  memory_probe=None, timer=perf_counter):
    rows = tuple(tuple(row) for row in rows)
    if (not callable(predict) or type(warmup) is not int or warmup < 0
            or type(repetitions) is not int or repetitions < 1
            or type(batch_size) is not int or batch_size < 1 or len(rows) != batch_size
            or not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows)
            or type(thread_reference) is not str or not thread_reference.strip()):
        raise InputContractError("explicit_benchmark_configuration_required")
    for row in rows:
        _numbers(row)
    if serialized_model_bytes is not None and (type(serialized_model_bytes) is not int or serialized_model_bytes < 0):
        raise InputContractError("invalid_serialized_model_size")
    if not callable(timer) or (memory_probe is not None and not callable(memory_probe)):
        raise InputContractError("benchmark_callbacks_required")
    latencies, memory = [], []
    try:
        for _ in range(warmup):
            predictions = predict(rows)
            if not isinstance(predictions, tuple) or len(predictions) != batch_size:
                raise InputContractError("benchmark_prediction_width_mismatch")
        for _ in range(repetitions):
            start = timer()
            predictions = predict(rows)
            stop = timer()
            _numbers((start, stop))
            elapsed = stop - start
            if elapsed <= 0:
                raise InputContractError("positive_benchmark_duration_required")
            if not isinstance(predictions, tuple) or len(predictions) != batch_size:
                raise InputContractError("benchmark_prediction_width_mismatch")
            latencies.append(elapsed)
            if memory_probe is not None:
                value = memory_probe()
                if type(value) is not int or value < 0:
                    raise InputContractError("invalid_memory_probe")
                memory.append(value)
    except InputContractError:
        raise
    except Exception:
        raise InputContractError("synthetic_benchmark_callback_failed") from None
    return {"result_category": "SYNTHETIC RESULT", "boundary": "trusted_model_only_predict_callback",
            "warmup": warmup, "repetitions": repetitions, "batch_size": batch_size,
            "thread_reference": thread_reference, "thread_configuration_authenticated": False,
            "p50_batch_ms": _quantile(latencies, .5) * 1000,
            "p95_batch_ms": _quantile(latencies, .95) * 1000,
            "throughput_artifacts_per_second": batch_size * repetitions / sum(latencies),
            "timed_batch_seconds": latencies,
            "measured_memory_bytes_max": max(memory) if memory else None,
            "memory_probe_scope": "caller_supplied_not_authenticated" if memory else "unavailable",
            "serialized_model_bytes": serialized_model_bytes,
            "selection_performed": False}

