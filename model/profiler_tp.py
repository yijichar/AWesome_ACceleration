import time
from collections import defaultdict, deque
from contextlib import contextmanager
import torch


class TPProfiler:
    """
    轻量级 profiler:
    - 支持 CPU wall time
    - 支持 CUDA synchronize 后计时（更准，但有扰动）
    - 聚合统计 sum/count/avg/max
    """
    def __init__(self, enabled: bool = False, cuda_sync: bool = False, keep_last: int = 0):
        self.enabled = enabled
        self.cuda_sync = cuda_sync
        self.stats = defaultdict(lambda: {"sum": 0.0, "count": 0, "max": 0.0})
        self.last = defaultdict(lambda: deque(maxlen=keep_last)) if keep_last > 0 else None

    def reset(self):
        self.stats.clear()
        if self.last is not None:
            self.last.clear()

    def _sync(self):
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            dt = time.perf_counter() - t0
            s = self.stats[name]
            s["sum"] += dt
            s["count"] += 1
            if dt > s["max"]:
                s["max"] = dt
            if self.last is not None:
                self.last[name].append(dt)

    def add(self, name: str, dt: float):
        if not self.enabled:
            return
        s = self.stats[name]
        s["sum"] += dt
        s["count"] += 1
        if dt > s["max"]:
            s["max"] = dt
        if self.last is not None:
            self.last[name].append(dt)

    def summary(self, prefix: str = "") -> str:
        if not self.enabled:
            return f"{prefix}TPProfiler disabled"
        lines = [f"{prefix}TPProfiler summary:"]
        for k in sorted(self.stats.keys()):
            s = self.stats[k]
            avg = s["sum"] / s["count"] if s["count"] else 0.0
            lines.append(
                f"{prefix}{k:32s}  count={s['count']:6d}  "
                f"sum={s['sum']*1000:10.3f} ms  avg={avg*1000:9.3f} ms  max={s['max']*1000:9.3f} ms"
            )
        return "\n".join(lines)

    def get(self):
        return self.stats
    
    def summary_filtered(self, include_prefix: str = "", prefix: str = "") -> str:
        if not self.enabled:
            return f"{prefix}TPProfiler disabled"
        lines = [f"{prefix}TPProfiler summary:"]
        for k in sorted(self.stats.keys()):
            if include_prefix and not k.startswith(include_prefix):
                continue
            s = self.stats[k]
            avg = s["sum"] / s["count"] if s["count"] else 0.0
            lines.append(
                f"{prefix}{k:40s}  count={s['count']:6d}  "
                f"sum={s['sum']*1000:10.3f} ms  avg={avg*1000:9.3f} ms  max={s['max']*1000:9.3f} ms"
            )
        return "\n".join(lines)