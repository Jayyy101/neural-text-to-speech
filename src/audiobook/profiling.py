"""Best-effort, process-local timing trace outside audiobook run state.

TTS_PROFILE=1 plus TTS_PROFILE_DIR opt a CLI process in. The UI calls activate() for
each job. A bounded daemon queue keeps trace I/O off production threads;
events may be lost if tracing cannot keep up or the process exits abruptly.
"""

import json
import os
from pathlib import Path
import queue
import sys
import threading
import time


MAX_PENDING_EVENTS = 4096


class _NoopSpan:
    span_id = None
    active = False

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        return False

    def add(self, **metadata):
        pass

    def finish(self, outcome="ok", **metadata):
        pass


_NOOP = _NoopSpan()
_directory = None
_failed = False
_counter = 0
_writer = None
_pending = None
try:
    _local = threading.local()
    _lock = threading.Lock()
    _process_uid = f"{sys.platform}:{os.getpid()}:{time.time_ns()}"
except Exception:
    _local = None
    _lock = None
    _process_uid = "unavailable"
    _failed = True


def _failure(error):
    """Disable tracing without performing potentially blocking diagnostic I/O."""
    global _failed
    try:
        _failed = True
    except Exception:
        pass


def _writer_failure(error, pending):
    """An old job's delayed writer failure cannot disable a newer job."""
    try:
        if _pending is pending:
            _failure(error)
    except Exception:
        pass


def enabled():
    try:
        return (_directory is not None and not _failed and _writer is not None
                and _writer.is_alive())
    except Exception:
        return False


def _write_event(destination, event, state):
    """Called only by the daemon writer; close each file after writing."""
    if state["destination"] != destination:
        state["destination"] = destination
        destination.mkdir(parents=True, exist_ok=True)
        state["part"] += 1
        state["path"] = destination / (
            f"events-{sys.platform}-{os.getpid()}-{state['part']:05d}.jsonl"
        )
        mode = "x"
    else:
        mode = "a"
    with state["path"].open(mode, encoding="utf-8") as output:
        output.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")


def _run_writer(pending, stop):
    state = {"destination": None, "path": None, "part": 0}
    try:
        while True:
            try:
                destination, event = pending.get(timeout=0.1)
            except queue.Empty:
                if stop.is_set():
                    break
                continue
            if _failed:
                continue
            try:
                _write_event(destination, event, state)
            except Exception as error:
                _writer_failure(error, pending)
                break
    except Exception as error:
        _writer_failure(error, pending)


def activate(directory):
    """Reset tracing for a UI job; None selects the disabled path."""
    global _directory, _failed, _writer, _pending, _counter
    try:
        previous = _writer
        if previous is not None:
            previous.stop.set()
        _directory = None
        _writer = None
        _pending = None
        _counter = 0
        _failed = False
        if directory is None:
            return
        if _local is None or _lock is None:
            _failed = True
            return
        destination = Path(directory)
        pending = queue.Queue(maxsize=MAX_PENDING_EVENTS)
        stop = threading.Event()
        writer = threading.Thread(target=_run_writer, args=(pending, stop),
                                  name="audiobook-profile-writer", daemon=True)
        writer.stop = stop
        writer.start()
        _pending = pending
        _writer = writer
        _directory = destination
    except Exception as error:
        _directory = None
        _writer = None
        _pending = None
        _failure(error)


def _stack():
    stack = getattr(_local, "stack", None)
    if stack is None:
        stack = []
        _local.stack = stack
    return stack


class Span:
    def __init__(self, name, parent_id=None, **metadata):
        self.name = name
        self.parent_id = parent_id
        self.metadata = metadata
        self.span_id = None
        self.started_ns = None
        self.thread_id = None
        self.active = False

    def __enter__(self):
        if not enabled():
            return self
        try:
            stack = _stack()
            parent = stack[-1] if stack else None
            with _lock:
                global _counter
                _counter += 1
                self.span_id = f"{_process_uid}:{_counter}"
            self.parent_id = self.parent_id or (parent.span_id if parent else None)
            if parent:
                self.metadata = {
                    **{key: value for key, value in parent.metadata.items()
                       if key in {"unit_id", "attempt_id", "request_id"}},
                    **self.metadata,
                }
            self.thread_id = threading.get_ident()
            self.started_ns = time.perf_counter_ns()
            stack.append(self)
            self.active = True
        except Exception as error:
            _failure(error)
        return self

    def add(self, **metadata):
        if self.active:
            try:
                self.metadata.update(metadata)
            except Exception as error:
                _failure(error)

    def finish(self, outcome="ok", **metadata):
        if not self.active:
            return
        self.active = False
        try:
            stack = _stack()
            if stack and stack[-1] is self:
                stack.pop()
            if not enabled():
                return
            ended_ns = time.perf_counter_ns()
            self.metadata.update(metadata)
            event = {
                "schema": 1, "name": self.name, "span_id": self.span_id,
                "parent_id": self.parent_id, "pid": os.getpid(),
                "process_uid": _process_uid,
                "thread_id": self.thread_id, "start_ns": self.started_ns,
                "end_ns": ended_ns, "duration_ns": ended_ns - self.started_ns,
                "outcome": outcome, "metadata": dict(self.metadata),
            }
            _pending.put_nowait((_directory, event))
        except queue.Full:
            pass
        except Exception as error:
            _failure(error)

    def __exit__(self, kind, error, traceback):
        try:
            self.finish("error" if kind is not None else "ok",
                        **({"error_type": kind.__name__} if kind is not None else {}))
        except Exception as profiling_error:
            _failure(profiling_error)
        return False


def span(name, *, parent_id=None, **metadata):
    if not enabled():
        return _NOOP
    try:
        return Span(name, parent_id=parent_id, **metadata)
    except Exception as error:
        _failure(error)
        return _NOOP


def begin(name, *, parent_id=None, **metadata):
    try:
        return span(name, parent_id=parent_id, **metadata).__enter__()
    except Exception as error:
        _failure(error)
        return _NOOP


def mark(name, **metadata):
    try:
        with span(name, **metadata):
            pass
    except Exception as error:
        _failure(error)


def annotate(**metadata):
    if enabled():
        try:
            stack = _stack()
            if stack:
                stack[-1].add(**metadata)
        except Exception as error:
            _failure(error)


def flush():
    """Compatibility hook: records are already queued for the daemon writer."""
    return None


try:
    _environment_directory = (os.environ.get("TTS_PROFILE_DIR")
                              if os.environ.get("TTS_PROFILE") == "1" else None)
    activate(_environment_directory)
except Exception as error:
    _failure(error)
