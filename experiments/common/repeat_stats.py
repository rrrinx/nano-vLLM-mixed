"""Statistics across complete replays, without discarding raw measurements."""

import math
import statistics


def summarize_repeats(values, trim_fraction):
    values = sorted(value for value in values if value is not None and math.isfinite(value))
    count = len(values)
    trimmed = math.floor(count * trim_fraction)
    mean = statistics.fmean(values) if count else None
    std = statistics.stdev(values) if count > 1 else None
    return {
        "n": count,
        "mean": mean,
        "trimmed_mean": statistics.fmean(values[trimmed:count - trimmed]) if count else None,
        "median": statistics.median(values) if count else None,
        "std": std,
        "cv_pct": 100 * std / abs(mean) if std is not None and mean else None,
        "min": values[0] if count else None,
        "max": values[-1] if count else None,
        "trimmed_each_tail": trimmed,
    }
