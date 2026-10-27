"""Measure what one dropped CaltopoSession retains. Run by hand.

Not collected by pytest: this is the experiment that established the
numbers in the MapReader design (2026-09-07), kept so they can be
re-measured against a new library version. No network is touched.
"""

import gc
import logging
import os
import tempfile
import threading
import tracemalloc

logging.disable(logging.CRITICAL)
from caltopo_python import CaltopoSession  # noqa: E402

ini = os.path.join(tempfile.mkdtemp(), "cts.ini")
with open(ini, "w", encoding="utf-8") as handle:
    handle.write("[you]\nid=x\nkey=y\naccountId=z\n")


def fake_setup(self):
    return True


def fake_open(self, mapID="", **kwargs):
    self.mapData = {"ids": {}, "state": {"features": [
        {"id": f"f{i}", "properties": {"class": "Shape", "title": "t" * 20},
         "geometry": {"type": "LineString",
                      "coordinates": [[-122.3 + k * 1e-5, 37.4 + k * 1e-5]
                                      for k in range(400)]}}
        for i in range(85)]}}
    return True


CaltopoSession._setupSession = fake_setup
CaltopoSession.openMap = fake_open

N = 20
gc.collect()
tracemalloc.start()
threads_before = threading.active_count()
heap_before = tracemalloc.get_traced_memory()[0]
for _ in range(N):
    session = CaltopoSession("caltopo.com", "ABC", configpath=ini, account="you",
                             sync=False)
    session = None
gc.collect()
threads_after = threading.active_count()
heap_after = tracemalloc.get_traced_memory()[0]
alive = sum(1 for o in gc.get_objects() if isinstance(o, CaltopoSession))

print(f"sessions opened and dropped: {N}")
print(f"threads: {threads_before} -> {threads_after} (+{threads_after - threads_before})")
print(f"heap: +{(heap_after - heap_before) / 1e6 / N:.2f} MB per dropped session")
print(f"CaltopoSession objects still alive: {alive}")
