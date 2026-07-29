from __future__ import annotations

import math
import statistics


def distribution(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0}

    def percentile(value):
        position = (len(ordered) - 1) * value / 100.0
        lower = int(math.floor(position))
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    return {"count": len(ordered), "mean": sum(ordered) / len(ordered),
            "stddev": statistics.pstdev(ordered), "minimum": ordered[0],
            "p50": percentile(50), "p95": percentile(95),
            "p99": percentile(99),
            "maximum": ordered[-1]}


class Metrics:
    def __init__(self):
        self.requests = []

    def record(self, **values):
        item = {"request_index": len(self.requests), **values}
        self.requests.append(item)
        return item

    def summary(self, fields):
        return {field: distribution(
            item[field] for item in self.requests if field in item)
                for field in fields}

    def reset(self):
        self.requests.clear()
