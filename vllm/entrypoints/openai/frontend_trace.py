# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lightweight frontend request tracing for local performance debugging.

Enabled only when VLLM_FRONTEND_TRACE_DIR or VLLM_FRONTEND_TRACE_FILE is set.
It records requests that have entered the OpenAI frontend but have not yet been
submitted to EngineCore. This is intentionally file-based so it can be used on
ad-hoc benchmark hosts without adding new metrics plumbing.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_TRACE: "FrontendTrace | None" = None
_TRACE_LOCK = threading.Lock()


class FrontendTrace:
    def __init__(self) -> None:
        trace_file = os.environ.get("VLLM_FRONTEND_TRACE_FILE")
        trace_dir = os.environ.get("VLLM_FRONTEND_TRACE_DIR")
        self.enabled = bool(trace_file or trace_dir)
        self._active: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._started_summary = False
        self._pid = os.getpid()
        self._summary_interval_s = float(
            os.environ.get("VLLM_FRONTEND_TRACE_INTERVAL_S", "5")
        )
        self._stuck_s = tuple(
            float(item)
            for item in os.environ.get(
                "VLLM_FRONTEND_TRACE_STUCK_S", "1,5,10,30,60"
            ).split(",")
            if item
        )
        self._log_summary = os.environ.get("VLLM_FRONTEND_TRACE_LOG", "1") != "0"

        if not self.enabled:
            self._path: Path | None = None
            return

        if trace_file:
            self._path = Path(trace_file)
        else:
            assert trace_dir is not None
            self._path = Path(trace_dir) / f"frontend_trace_{self._pid}.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._emit_unlocked(
            {
                "event": "trace_start",
                "summary_interval_s": self._summary_interval_s,
                "stuck_s": self._stuck_s,
            }
        )
        self._start_summary_thread()

    def request_start(
        self,
        request_id: str,
        *,
        endpoint: str,
        stream: bool | None,
        max_tokens: int | None,
    ) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        wall = time.time()
        with self._lock:
            self._active[request_id] = {
                "start_perf": now,
                "start_time": wall,
                "endpoint": endpoint,
                "stream": stream,
                "max_tokens": max_tokens,
            }
            self._emit_unlocked(
                {
                    "event": "frontend_start",
                    "request_id": request_id,
                    "endpoint": endpoint,
                    "stream": stream,
                    "max_tokens": max_tokens,
                    "active_frontend": len(self._active),
                }
            )

    def stage(self, request_id: str, stage: str, **fields: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            active = self._active.get(request_id)
            if active is not None:
                active[stage + "_perf"] = time.perf_counter()
            self._emit_unlocked(
                {
                    "event": stage,
                    "request_id": request_id,
                    "active_frontend": len(self._active),
                    **fields,
                }
            )

    def engine_add(self, request_id: str) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        with self._lock:
            active = self._active.pop(request_id, None)
            row: dict[str, Any] = {
                "event": "engine_add",
                "request_id": request_id,
                "active_frontend": len(self._active),
            }
            if active is not None:
                row["frontend_s"] = now - active["start_perf"]
                render_perf = active.get("render_done_perf")
                if isinstance(render_perf, float):
                    row["post_render_s"] = now - render_perf
            self._emit_unlocked(row)

    def finish(self, request_id: str, reason: str, **fields: Any) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        with self._lock:
            active = self._active.pop(request_id, None)
            row: dict[str, Any] = {
                "event": "frontend_finish",
                "request_id": request_id,
                "reason": reason,
                "active_frontend": len(self._active),
                **fields,
            }
            if active is not None:
                row["frontend_s"] = now - active["start_perf"]
            self._emit_unlocked(row)

    def _start_summary_thread(self) -> None:
        if self._started_summary or self._summary_interval_s <= 0:
            return
        self._started_summary = True
        thread = threading.Thread(
            target=self._summary_loop,
            name="vllm-frontend-trace",
            daemon=True,
        )
        thread.start()

    def _summary_loop(self) -> None:
        while True:
            time.sleep(self._summary_interval_s)
            self.summary()

    def summary(self) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        with self._lock:
            ages = [now - item["start_perf"] for item in self._active.values()]
            row: dict[str, Any] = {
                "event": "frontend_summary",
                "active_frontend": len(self._active),
                "oldest_s": max(ages) if ages else 0.0,
            }
            for threshold in self._stuck_s:
                row[f"over_{threshold:g}s"] = sum(age >= threshold for age in ages)
            oldest = sorted(
                (
                    (now - item["start_perf"], request_id, item)
                    for request_id, item in self._active.items()
                ),
                reverse=True,
            )[:10]
            row["oldest_requests"] = [
                {
                    "request_id": request_id,
                    "age_s": age,
                    "endpoint": item.get("endpoint"),
                    "stream": item.get("stream"),
                    "max_tokens": item.get("max_tokens"),
                }
                for age, request_id, item in oldest
            ]
            self._emit_unlocked(row)

        if self._log_summary and row["active_frontend"]:
            logger.info(
                "Frontend trace: active=%d oldest=%.3fs over_1s=%s over_5s=%s "
                "over_10s=%s over_30s=%s over_60s=%s",
                row["active_frontend"],
                row["oldest_s"],
                row.get("over_1s", 0),
                row.get("over_5s", 0),
                row.get("over_10s", 0),
                row.get("over_30s", 0),
                row.get("over_60s", 0),
            )

    def _emit_unlocked(self, row: dict[str, Any]) -> None:
        if self._path is None:
            return
        row.setdefault("time", time.time())
        row.setdefault("pid", self._pid)
        with self._path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, sort_keys=True) + "\n")


def get_frontend_trace() -> FrontendTrace:
    global _TRACE
    if _TRACE is not None:
        return _TRACE
    with _TRACE_LOCK:
        if _TRACE is None:
            _TRACE = FrontendTrace()
    return _TRACE
