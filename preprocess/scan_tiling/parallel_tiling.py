import logging
from typing import Sequence
import multiprocessing as mp
from pathlib import Path
import tempfile

from tqdm import tqdm  # type: ignore

from corresponding import Case
from configuration import Configuration
from process import copy_process, tile_process

log = logging.getLogger()


def process_monitor(total_cases: int, msg_queue: mp.Queue, include_copy: bool):
    if include_copy:
        copy_pbar = tqdm(
            total=total_cases,
            position=0,
            dynamic_ncols=True,
            disable=(total_cases == 1),
        )
        tile_pbar = tqdm(
            total=total_cases,
            position=1,
            dynamic_ncols=True,
            disable=(total_cases == 1),
        )
    else:
        tile_pbar = tqdm(
            total=total_cases,
            position=0,
            dynamic_ncols=True,
            disable=(total_cases == 1),
        )

    while True:
        msg = msg_queue.get()

        if isinstance(msg, str):
            # copy_pbar.write(msg)
            tile_pbar.write(msg)
        else:
            if msg["source"] == "COPY-WORKER":
                if msg["msg"] is not None:
                    copy_pbar.write(msg["msg"])
                if msg["update"]:
                    copy_pbar.update()
            elif msg["source"] == "TILE-WORKER":
                if msg["msg"] is not None:
                    tile_pbar.write(msg["msg"])
                if msg["update"]:
                    tile_pbar.update()


def tile_cases(cases: Sequence[Case], conf: Configuration):
    if len(cases) > 1:
        log.info("Started parallel tiling")
    else:
        log.info("Started tiling")

    total_cases = len(cases)

    copy_workers = []
    tile_workers = []
    try:
        # Parent progress bar
        progress_queue: mp.Queue = mp.Queue()
        pm_worker = mp.Process(
            target=process_monitor,
            args=(total_cases, progress_queue, conf.use_cache),
            daemon=True,
        )
        pm_worker.start()

        # Input queue
        input_queue: mp.JoinableQueue = mp.JoinableQueue()

        # Max sizes to have in queue
        max_cases = 3 * conf.tile_workers if conf.use_cache else 0
        case_queue: mp.JoinableQueue = mp.JoinableQueue(maxsize=max_cases)

        if conf.use_cache:
            conf.cache_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(conf.cache_root)) as tmp_cache_dir:
            if conf.use_cache:
                log.info("Starting copying to: {}".format(tmp_cache_dir))

                # Calculate the maximum number of workers needed
                num_copy_workers = min(conf.copy_workers, total_cases)
                log.info("Using {} copy workers.".format(num_copy_workers))

                # Enqueue all cases for copying
                for sample in cases:
                    input_queue.put(sample)

                # Submit copy cases
                for worker_id in range(num_copy_workers):
                    progress_queue.put("Starting copy worker: {}".format(worker_id))
                    copy_worker = mp.Process(
                        target=copy_process,
                        args=(
                            worker_id,
                            input_queue,
                            case_queue,
                            progress_queue,
                            Path(tmp_cache_dir),
                            conf.verbose,
                        ),
                        daemon=True,
                    )
                    copy_worker.start()
                    copy_workers.append(copy_worker)

                    # Add None to end of input_queue for each copy worker to signal the
                    # end of the queue
                    input_queue.put(None)
            else:
                # If we do not use the cache we directly enqueue all cases
                for sample in cases:
                    case_queue.put(sample)

            # Output queue
            output_queue: mp.JoinableQueue = mp.JoinableQueue()

            # Determine number of process workers
            num_tile_workers = min(conf.tile_workers, total_cases)

            # Start tiling workers
            for worker_id in range(num_tile_workers):
                progress_queue.put("Starting tile worker: {}".format(worker_id))
                tile_worker = mp.Process(
                    target=tile_process,
                    args=(
                        worker_id,
                        case_queue,
                        output_queue,
                        progress_queue,
                        conf,
                    ),
                    daemon=True,
                )
                tile_worker.start()
                tile_workers.append(tile_worker)

            # If using caching, we wait until all scans have been copied
            if conf.use_cache:
                progress_queue.put("Main - Waiting for input queue to empty")
                input_queue.join()
                progress_queue.put("Copying of scans completed")

            # Here we signal all tile workers that work is finished
            for _ in range(num_tile_workers):
                case_queue.put(None)

            # Wait for all tile_threads to finish
            case_queue.join()
            progress_queue.put("Main - Tiling is complete!")
            pm_worker.terminate()

            # Cleanup cache directory if using cache
            if conf.use_cache:
                log.info(f"\nRemoving temporary dir: {tmp_cache_dir}")

        # Calculate total
        results = []
        for i, _ in enumerate(range(total_cases)):
            _, tiles = output_queue.get()
            results.append(tiles)

        # Check number of produced tiles
        total_num_tiles = sum(results)
        log.info("Created: %d tiles" % total_num_tiles)

    except KeyboardInterrupt:
        log.info("Keyboard interrupt received")
        log.info(f"Stopping {len(copy_workers)} copy processes workers")
        for copy_worker in copy_workers:
            copy_worker.terminate()

        log.info(f"Stopping {len(tile_workers)} tile workers.")
        for tile_worker in tile_workers:
            tile_worker.terminate()

        pm_worker.terminate()

    except Exception as e:
        log.error(f"Exception during tiling:\n\t{e}")
