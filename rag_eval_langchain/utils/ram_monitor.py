"""
RAM usage monitor with background sampling and per-stage tracking.

Usage:
    monitor = RamMonitor(log_path="ram_log.csv", interval=2)
    monitor.start()

    with monitor.stage("my_stage"):
        # expensive work here

    monitor.stop()
"""

import csv
import threading
import time
from contextlib import contextmanager
from datetime import datetime

import psutil


class RamMonitor:
    def __init__(self, log_path: str = "ram_log.csv", interval: float = 2.0):
        self.log_path = log_path
        self.interval = interval
        self._process = psutil.Process()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._current_stage = "idle"
        self._lock = threading.Lock()

        # Write CSV header
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "stage", "rss_mb", "percent"])

    def start(self):
        self._thread.start()
        print(f"[RamMonitor] Started — logging to '{self.log_path}' every {self.interval}s")

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        rss = self._process.memory_info().rss / 1024 ** 2
        print(f"[RamMonitor] Stopped — final RAM: {rss:.1f} MB")

    @contextmanager
    def stage(self, name: str):
        rss_before = self._process.memory_info().rss / 1024 ** 2
        print(f"[RamMonitor] Stage '{name}' started  (RAM: {rss_before:.1f} MB)")
        with self._lock:
            self._current_stage = name
        try:
            yield
        finally:
            rss_after = self._process.memory_info().rss / 1024 ** 2
            delta = rss_after - rss_before
            sign = "+" if delta >= 0 else ""
            print(
                f"[RamMonitor] Stage '{name}' finished "
                f"(RAM: {rss_after:.1f} MB, delta: {sign}{delta:.1f} MB)"
            )
            with self._lock:
                self._current_stage = "idle"

    def _sample_loop(self):
        while not self._stop_event.wait(self.interval):
            mem = self._process.memory_info()
            rss_mb = mem.rss / 1024 ** 2
            percent = self._process.memory_percent()
            ts = datetime.now().strftime("%H:%M:%S")
            with self._lock:
                stage = self._current_stage
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow([ts, stage, f"{rss_mb:.1f}", f"{percent:.2f}"])
