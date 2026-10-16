from __future__ import annotations

import math
import time
from contextlib import contextmanager

STAGES = ("document_preparation", "query_embedding", "scoring", "selection", "render", "compression", "generation")


class StageTimer:
    """Nonoverlapping stage timer. Missing, skipped, and measured are distinct."""
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.stages = {name: {"status": "unmeasured", "seconds": None} for name in STAGES}
        self.active = None

    @contextmanager
    def measure(self, name):
        if name not in self.stages or self.active or self.stages[name]["status"] != "unmeasured":
            raise ValueError("Unknown, overlapping, or repeated timing stage")
        self.active = name
        start = self.clock()
        try:
            yield
        except BaseException:
            self.stages[name] = {"status": "failed", "seconds": None}
            raise
        else:
            self.stages[name] = {"status": "measured", "seconds": self.clock() - start}
        finally:
            self.active = None


def timing_record(*, document_preparation=None, query_embedding=None, scoring=None,
                  selection=None, render=None, compression=None, generation=None,
                  skipped=(), document_id=None, generation_mode="unmeasured"):
    values = locals()
    stages = {}
    for name in STAGES:
        value = values[name]
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("Invalid stage duration")
        stages[name] = {"status": "skipped" if name in skipped else "unmeasured" if value is None else "measured",
                        "seconds": 0.0 if name in skipped else value}
    return {"schema_version": "2", "stages": stages, "document_id": document_id,
            "document_cost_ownership": "shared_once_per_document", "generation_mode": generation_mode,
            "request_wall_seconds": None, "request_wall_status": "unmeasured_separate_build_and_generation"}


def stage_total(record, *, indexed=False):
    names = STAGES[1:] if indexed else STAGES
    stages = record["stages"]
    total = 0.0
    for name in names:
        stage = stages[name]
        if name == "render" and stage["status"] == "included_in_compression":
            if stages["compression"]["status"] != "measured":
                return None
            continue
        if stage["status"] not in {"measured", "skipped"} or stage["seconds"] is None:
            return None
        total += stage["seconds"]
    return total
