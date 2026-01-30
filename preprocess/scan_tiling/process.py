import os
import logging
import time
import multiprocessing as mp
from pathlib import Path
import shutil
from typing import Any, Optional, Tuple

from common import common_utils
from configuration import Configuration
from corresponding import Case
import perform_tiling

log = logging.getLogger()


def copy_process(
    worker_id: int,
    input_queue: mp.JoinableQueue,
    output_queue: mp.JoinableQueue,
    progress_queue: mp.Queue,
    cache_dir: Path,
    verbose: bool = False,
):
    def set_msg(msg: Optional[Any], update: bool = False):
        if msg is None:
            msg_status = None
        else:
            msg_status = "CopyWorker-{:02d} - {}".format(worker_id, msg)
        progress_queue.put(dict(source="COPY-WORKER", msg=msg_status, update=update))

    while True:
        try:
            sample = input_queue.get()

            # We place None at the end of the input queue to signal the end
            if sample is None:
                set_msg("Exiting")
                input_queue.task_done()
                return
            else:
                cache_path = cache_dir.joinpath(sample.relative_dir, sample.scan.name)
                cache_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy scan if file does not exist
                if not cache_path.exists():
                    common_utils.copy_with_rsync(sample.scan, cache_path, verbose)
                    if sample.scan.suffix == ".mrxs":
                        # In this case, also copy the corresponding scan folder with the
                        # same name and that are placed in the same folder as the scan
                        scan_stem = sample.scan.stem
                        source_scan_folder = sample.scan.parent.joinpath(scan_stem)
                        assert source_scan_folder.exists()
                        common_utils.copy_with_rsync(
                            source_scan_folder,
                            cache_path.parent,
                            verbose,
                        )
                    if verbose:
                        set_msg(f"Copied: {sample.scan.name} -> {cache_path.parent}")

                else:
                    set_msg(f"WARNING: Scan already exists in cache: {cache_path}")

                # Change the scan_filepath
                sample.scan = cache_path
                sample.remove_scan = True
                output_queue.put(sample)
                msg = None
                if verbose:
                    msg = f"Added {sample.scan.name} to queue."
                    msg += f"(~{output_queue.qsize()} items in queue, "
                    msg += f"{input_queue.qsize()} remaining)"

                # Increase progress bar
                set_msg(msg, update=True)

                # Mark the task as done
                input_queue.task_done()

        except Exception as e:
            set_msg(f"Error occurred during copying file: {sample.scan}\n{e}")


def run_tile(sample: Case, conf: Configuration) -> Tuple[Case, int]:
    # Set tile count
    tile_count = 0

    # Run tiling
    try:
        tile_count = perform_tiling.tile_scan(sample, conf)
    except Exception as e:
        log.warning(f"Caught exception: '{e}' in {sample.scan}")
        try:
            tile_count = perform_tiling.tile_scan(sample, conf, increase_level=True)
        except Exception:
            raise Exception("Tiling failed")

    return sample, int(tile_count)


def tile_process(
    worker_id: int,
    input_queue: mp.JoinableQueue,
    output_queue: mp.JoinableQueue,
    progress_queue: mp.Queue,
    conf: Configuration,
):
    def set_msg(msg: Any, update=False):
        msg_status = "TileWorker-{:02d} {}".format(worker_id, msg)
        progress_queue.put(dict(source="TILE-WORKER", msg=msg_status, update=update))

    while True:
        try:
            # set_msg("Getting next sample...")
            sample = input_queue.get()

            if sample is None:
                set_msg("Exiting")
                input_queue.task_done()
                return
            else:
                # set_msg("Starting processing: {}.".format(sample.scan))

                start = time.time()

                # Tile sample
                result = run_tile(sample, conf)
                _, num_tiles = result

                # Check if we should remove scan
                remove_sample = sample.remove_scan

                # Check file should be removed and the file exists
                if remove_sample and sample.scan.is_file():
                    if conf.verbose:
                        set_msg("Removing scan: {}".format(sample.scan))
                    # set_msg("TRIAL: NOT REMOVING")
                    os.unlink(sample.scan)
                    if sample.scan.suffix == ".mrxs":
                        mrxs_folder = sample.scan.parent.joinpath(sample.scan.stem)
                        assert mrxs_folder.is_dir()
                        shutil.rmtree(mrxs_folder)

                # Update status and increase progress
                elapsed = time.time() - start
                tps = int(num_tiles) / elapsed
                msg = "{:>100}{:>15}{:>15}{:>17}".format(
                    sample.scan.name,
                    f"{num_tiles:>6} tiles",
                    common_utils.format_time(elapsed),
                    f"{tps:>6.3f} tiles/s",
                )
                set_msg(msg, update=True)

                # Return result
                output_queue.put(result)

            # Notify task done
            input_queue.task_done()

        except Exception as e:
            set_msg("Got an exception")
            set_msg("Exception: {}".format(e))
            output_queue.put((sample, 0))
            raise e
