from __future__ import annotations

import statistics
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import pynvml
except ImportError as exc:
    pynvml = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass
class PowerSample:
    monotonic_sec: float
    unix_sec: float
    power_w: float
    gpu_util_pct: int
    memory_util_pct: int
    temperature_c: int
    memory_used_mb: float


def _interpolate(samples: List[PowerSample], t: float) -> float:
    if t <= samples[0].monotonic_sec:
        return samples[0].power_w
    if t >= samples[-1].monotonic_sec:
        return samples[-1].power_w
    for left, right in zip(samples, samples[1:]):
        if left.monotonic_sec <= t <= right.monotonic_sec:
            span = right.monotonic_sec - left.monotonic_sec
            if span <= 0:
                return right.power_w
            ratio = (t - left.monotonic_sec) / span
            return left.power_w + ratio * (right.power_w - left.power_w)
    return samples[-1].power_w


def integrate_power(
    samples: List[PowerSample],
    start: Optional[float] = None,
    end: Optional[float] = None,
    baseline_w: float = 0.0,
) -> float:
    if len(samples) < 2:
        return 0.0
    start_t = samples[0].monotonic_sec if start is None else start
    end_t = samples[-1].monotonic_sec if end is None else end
    if end_t <= start_t:
        return 0.0

    points: List[Tuple[float, float]] = [
        (start_t, _interpolate(samples, start_t))
    ]
    points.extend(
        (sample.monotonic_sec, sample.power_w)
        for sample in samples
        if start_t < sample.monotonic_sec < end_t
    )
    points.append((end_t, _interpolate(samples, end_t)))

    joules = 0.0
    for (t0, p0), (t1, p1) in zip(points, points[1:]):
        adjusted_0 = max(p0 - baseline_w, 0.0)
        adjusted_1 = max(p1 - baseline_w, 0.0)
        joules += 0.5 * (adjusted_0 + adjusted_1) * (t1 - t0)
    return joules


class NVMLPowerMonitor:
    def __init__(self, gpu_index: int, sample_interval_sec: float):
        if pynvml is None:
            raise RuntimeError(
                "Install nvidia-ml-py before running the experiment"
            ) from _IMPORT_ERROR
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.sample_interval_sec = sample_interval_sec
        self.samples: List[PowerSample] = []
        self.baseline_w = 0.0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def device_info(self) -> Dict[str, Any]:
        name = pynvml.nvmlDeviceGetName(self.handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode("utf-8", errors="replace")
        uuid = pynvml.nvmlDeviceGetUUID(self.handle)
        if isinstance(uuid, bytes):
            uuid = uuid.decode("utf-8", errors="replace")
        return {"name": name, "uuid": str(uuid), "driver_version": driver}

    def _sample(self) -> PowerSample:
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return PowerSample(
            monotonic_sec=time.perf_counter(),
            unix_sec=time.time(),
            power_w=pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0,
            gpu_util_pct=int(util.gpu),
            memory_util_pct=int(util.memory),
            temperature_c=int(
                pynvml.nvmlDeviceGetTemperature(
                    self.handle, pynvml.NVML_TEMPERATURE_GPU
                )
            ),
            memory_used_mb=mem.used / (1024.0 * 1024.0),
        )

    def measure_baseline(self, duration_sec: float) -> Dict[str, Any]:
        values: List[PowerSample] = []
        deadline = time.perf_counter() + duration_sec
        while time.perf_counter() < deadline:
            values.append(self._sample())
            time.sleep(self.sample_interval_sec)
        if not values:
            values.append(self._sample())
        powers = [sample.power_w for sample in values]
        self.baseline_w = float(statistics.median(powers))
        return {
            "baseline_w_median": self.baseline_w,
            "baseline_w_mean": float(statistics.fmean(powers)),
            "baseline_w_min": min(powers),
            "baseline_w_max": max(powers),
            "baseline_samples": len(values),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self._sample())
            self._stop.wait(self.sample_interval_sec)

    def start(self) -> float:
        if self._thread is not None:
            raise RuntimeError("Monitor already running")
        self.samples = [self._sample()]
        self._stop.clear()
        start_time = self.samples[0].monotonic_sec
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return start_time

    def stop(self) -> Dict[str, Any]:
        if self._thread is None:
            raise RuntimeError("Monitor not running")
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.sample_interval_sec * 5))
        self.samples.append(self._sample())
        self._thread = None
        powers = [sample.power_w for sample in self.samples]
        return {
            "sample_count": len(self.samples),
            "sample_interval_sec": self.sample_interval_sec,
            "duration_sec": (
                self.samples[-1].monotonic_sec
                - self.samples[0].monotonic_sec
            ),
            "baseline_w": self.baseline_w,
            "total_energy_j": integrate_power(self.samples),
            "dynamic_energy_j": integrate_power(
                self.samples, baseline_w=self.baseline_w
            ),
            "mean_power_w": float(statistics.fmean(powers)),
            "median_power_w": float(statistics.median(powers)),
            "min_power_w": min(powers),
            "max_power_w": max(powers),
        }

    def interval_energy(
        self, start_monotonic: float, end_monotonic: float
    ) -> Dict[str, float]:
        return {
            "total_energy_j": integrate_power(
                self.samples, start=start_monotonic, end=end_monotonic
            ),
            "dynamic_energy_j": integrate_power(
                self.samples,
                start=start_monotonic,
                end=end_monotonic,
                baseline_w=self.baseline_w,
            ),
        }

    def serialized_samples(self) -> List[Dict[str, Any]]:
        return [asdict(sample) for sample in self.samples]

    def close(self) -> None:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
