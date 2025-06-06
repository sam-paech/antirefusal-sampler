from __future__ import annotations
import queue, threading, time, logging
import concurrent.futures
from typing import Tuple, List
from .refusal_detector import RefusalDetector      # relative import

logger = logging.getLogger(__name__)

class RefusalWorker:
    """
    A single-threaded service that batches up calls to RefusalDetector.

    * max_batch   – fire the detector as soon as this many items are queued
    * max_wait_ms – …or after this long since the first item arrived
    """

    def __init__(
        self,
        detector: RefusalDetector,
        max_batch: int = 8,
        max_wait_ms: int = 5,
    ) -> None:
        self.detector   = detector
        self.max_batch  = max_batch
        self.max_wait   = max_wait_ms / 1000.0
        self.q: queue.Queue[Tuple[str, str, concurrent.futures.Future]] = queue.Queue()
        self._stop = threading.Event()
        self._thr  = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    # ––––– public –––––
    def submit(self, user_txt: str, assistant_txt: str
               ) -> concurrent.futures.Future:
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self.q.put((user_txt, assistant_txt, fut))
        return fut

    def shutdown(self) -> None:
        self._stop.set()
        self._thr.join()

    # ––––– private –––––
    def _loop(self) -> None:
        queue_at_start = self.q.qsize() + 1   # +1 for the ‘first’ item we just took
        while not self._stop.is_set():
            try:
                first = self.q.get(timeout=0.1)          # block for the first job
            except queue.Empty:
                continue

            batch  = [first]
            t0     = time.perf_counter()

            # collect more until either cap or timeout
            while len(batch) < self.max_batch:
                remaining = self.max_wait - (time.perf_counter() - t0)
                if remaining <= 0:
                    break
                try:
                    batch.append(self.q.get(timeout=remaining))
                except queue.Empty:
                    break

            pairs: List[Tuple[str, str]] = [(u, a) for u, a, _ in batch]
            t1 = time.perf_counter()
            results = self.detector.is_refusal_batch(pairs)
            duration_ms = (time.perf_counter() - t1) * 1000
            logger.info(
                "RefusalWorker ran batch: size=%d  queue_at_start=%d  time=%.1f ms",
                len(batch), queue_at_start, duration_ms
            )

            for (_, _, fut), (is_ref, conf, lbl) in zip(batch, results):
                fut.set_result((is_ref, conf, lbl))
