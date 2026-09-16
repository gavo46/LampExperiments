"""
Opt-in instrumentation for per-utterance latency breakdown,
face-detection engagement stats, and
process CPU/RSS usage, plus a consolidated summary on exit.

Everything here is gated behind config.MEASURE. When it's False, every
public entry point is a cheap no-op (start_utterance() returns None,
start() does nothing), so nothing in this module changes program
behavior or adds meaningful overhead - it only observes. Flip
config.MEASURE to False to turn all of it off cleanly.
"""

import atexit
import statistics
import threading
import time

import psutil

import config

_lock = threading.Lock()

# --- latency: speech-end -> transcription -> LLM reply -> speech start ---

_LATENCY_STAGES = ("whisper_ms", "llm_ms", "tts_start_ms")
_latency_samples = {stage: [] for stage in _LATENCY_STAGES}
_utterance_count = 0


class UtteranceTrace:
    """
    Follows one utterance through the pipeline, collecting wall-clock
    marks so each stage's duration can be reported once it's done.

    Only ever created by start_utterance() when config.MEASURE is
    True - every call site elsewhere does `if trace: trace.mark(...)`,
    so passing None around when measurement is off costs nothing and
    changes nothing.
    """

    def __init__(self):
        self.marks = {"speech_end": time.monotonic()}

    def mark(self, name):
        self.marks[name] = time.monotonic()

    def finish(self):
        """
        Call once speech_start has been marked. Prints the per-stage
        breakdown for this utterance and folds it into the running
        samples for the final summary.
        """
        whisper_ms = _delta_ms(self.marks.get("speech_end"), self.marks.get("transcription_done"))
        llm_ms = _delta_ms(self.marks.get("transcription_done"), self.marks.get("llm_done"))
        tts_ms = _delta_ms(self.marks.get("llm_done"), self.marks.get("speech_start"))
        total_ms = _delta_ms(self.marks.get("speech_end"), self.marks.get("speech_start"))

        print(
            "[metrics] latency  "
            f"whisper={_fmt(whisper_ms)}  llm={_fmt(llm_ms)}  "
            f"tts_start={_fmt(tts_ms)}  total={_fmt(total_ms)}"
        )

        global _utterance_count
        with _lock:
            _utterance_count += 1
            for stage, value in zip(_LATENCY_STAGES, (whisper_ms, llm_ms, tts_ms)):
                if value is not None:
                    _latency_samples[stage].append(value)


def _delta_ms(start, end):
    if start is None or end is None:
        return None
    return (end - start) * 1000.0


def _fmt(value_ms):
    return f"{value_ms:.0f}ms" if value_ms is not None else "n/a"


def start_utterance():
    """
    Call at end-of-speech (speech/listen.py, right before handing audio
    to Whisper). Returns an UtteranceTrace to carry through the
    pipeline, or None if config.MEASURE is False.
    """
    if not config.MEASURE:
        return None
    return UtteranceTrace()


# --- engagement: face-detection flips/detection rate (vision/tracking.py) ---

_engagement_samples = []  # (detection_rate, flips_per_minute) per 30s window


def record_engagement(detection_rate, flips_per_minute):
    with _lock:
        _engagement_samples.append((detection_rate, flips_per_minute))


# --- resource usage: process CPU% / RSS, sampled periodically ---

_cpu_samples = []
_rss_samples = []


class _ResourceMonitor(threading.Thread):
    """
    Samples this process's CPU% and RSS every `sample_interval`
    seconds, and prints/records a mean+peak summary every
    `report_interval` seconds.
    """

    def __init__(self, sample_interval=2.0, report_interval=30.0):
        super().__init__(daemon=True)
        self.sample_interval = sample_interval
        self.report_interval = report_interval
        self._stop_event = threading.Event()
        self._process = psutil.Process()

    def run(self):
        self._process.cpu_percent()  # prime it; first reading is meaningless

        window_cpu = []
        window_rss = []
        last_report = time.monotonic()

        while not self._stop_event.wait(self.sample_interval):
            cpu = self._process.cpu_percent()
            rss_mb = self._process.memory_info().rss / (1024 * 1024)
            window_cpu.append(cpu)
            window_rss.append(rss_mb)

            if time.monotonic() - last_report >= self.report_interval:
                if window_cpu:
                    print(
                        "[metrics] resources  "
                        f"cpu_mean={statistics.mean(window_cpu):.1f}%  "
                        f"cpu_peak={max(window_cpu):.1f}%  "
                        f"rss_mean={statistics.mean(window_rss):.0f}MB  "
                        f"rss_peak={max(window_rss):.0f}MB"
                    )
                    with _lock:
                        _cpu_samples.extend(window_cpu)
                        _rss_samples.extend(window_rss)
                window_cpu = []
                window_rss = []
                last_report = time.monotonic()

    def stop(self):
        self._stop_event.set()


_resource_monitor = None


# --- lifecycle ---


def start():
    """
    Call once at startup (main.py). No-op unless config.MEASURE is
    True. Starts the resource-usage sampling thread and registers the
    final summary to print on exit (normal or Ctrl+C).
    """
    global _resource_monitor
    if not config.MEASURE:
        return
    _resource_monitor = _ResourceMonitor()
    _resource_monitor.start()
    atexit.register(print_summary)
    print("[metrics] instrumentation enabled (config.MEASURE = True)")


def print_summary():
    """Consolidated end-of-session report. Registered via atexit by
    start(), so it fires on clean exit without main.py needing to call
    it explicitly."""
    if not config.MEASURE:
        return

    with _lock:
        latency = {stage: list(samples) for stage, samples in _latency_samples.items()}
        utterance_count = _utterance_count
        engagement = list(_engagement_samples)
        cpu = list(_cpu_samples)
        rss = list(_rss_samples)

    print("\n" + "=" * 60)
    print("[metrics] SESSION SUMMARY")
    print("=" * 60)

    print(f"utterances handled: {utterance_count}")
    for stage in _LATENCY_STAGES:
        samples = latency[stage]
        if samples:
            print(
                f"  {stage}: median={statistics.median(samples):.0f}ms  "
                f"range=[{min(samples):.0f}, {max(samples):.0f}]ms  n={len(samples)}"
            )
        else:
            print(f"  {stage}: no samples")

    if engagement:
        detection_rates = [d for d, _ in engagement]
        flip_rates = [f for _, f in engagement]
        print(
            f"engagement: avg_detection_rate={statistics.mean(detection_rates):.1%}  "
            f"avg_flips_per_min={statistics.mean(flip_rates):.1f}  "
            f"(over {len(engagement)} x 30s window(s))"
        )
    else:
        print("engagement: no samples")

    if cpu:
        print(f"cpu: mean={statistics.mean(cpu):.1f}%  peak={max(cpu):.1f}%")
    else:
        print("cpu: no samples")

    if rss:
        print(f"memory (RSS): mean={statistics.mean(rss):.0f}MB  peak={max(rss):.0f}MB")
    else:
        print("memory (RSS): no samples")

    print("=" * 60)
