"""Lifetime containment for the workers owned by one BW pass."""

from __future__ import annotations

import multiprocessing
import os
import threading
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from multiprocessing.connection import wait
from multiprocessing.process import BaseProcess


def watch_bw_parent() -> None:
    """Exit a native worker if its multiprocessing parent disappears.

    CFFI releases the GIL during BW work, so this daemon can observe the public
    parent sentinel even while the worker is inside a long native update.
    """
    parent = multiprocessing.parent_process()
    if parent is None:
        raise RuntimeError("BW pool initializer requires a multiprocessing parent")

    def watch() -> None:
        wait([parent.sentinel])
        # Native work cannot be safely unwound from another thread. The entire
        # worker is disposable; incomplete shard artifacts are never reduced.
        os._exit(1)

    threading.Thread(target=watch, name="bw-parent-watch", daemon=True).start()


@contextmanager
def contained_bw_pool(pool: ProcessPoolExecutor) -> Iterator[ProcessPoolExecutor]:
    """Reap this executor's workers before propagating a failed pass.

    Python 3.11–3.13 have no public terminate_workers API. Capture only this
    executor's owned process handles, before shutdown clears its registry;
    never inspect or terminate unrelated multiprocessing children.
    """
    processes: tuple[BaseProcess, ...] = ()
    try:
        yield pool
        processes = tuple((getattr(pool, "_processes", None) or {}).values())
        pool.shutdown(wait=True)
    except BaseException:
        if not processes:
            processes = tuple((getattr(pool, "_processes", None) or {}).values())
        for process in processes:
            if process.is_alive():
                process.kill()
        pool.shutdown(wait=True, cancel_futures=True)
        for process in processes:
            process.join()
        raise
