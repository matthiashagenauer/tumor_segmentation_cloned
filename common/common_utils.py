import sys
import logging
from pathlib import Path
import shlex
import subprocess
from typing import List, Optional, Sequence, Tuple

log = logging.getLogger()


supported_scanners = {
    "aperio": ".svs",           # Aperio AT2
    "gt450dx": ".svs",          # Aperio GT450DX
    "xr": ".ndpi",              # NanoZoomer XR
    "s210": ".ndpi",            # NanoZoomer S210
    "s360": ".ndpi",            # NanoZoomer S360
    "mbm": ".svs",              # MBM
    "olympus": ".tiff",         # Olympus
    "philips": ".svs",          # Philips
    "3dhistech": ".mrxs",       # 3DHISTECH
    "mirax": ".mrxs",           # Catch-all .mrxs
    "unknown-tif": ".tif",      # Catch-all .tif
    "unknown-tiff": ".tiff",      # Catch-all .tiff
}


def plural_s(length: int) -> str:
    if length == 1:
        return ""
    else:
        return "s"


def format_time(total_seconds: float) -> str:
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    if hours > 0:
        formatted = f"{hours}h {minutes:>2}m {seconds:>2}s"
    elif minutes > 0:
        formatted = f"{minutes:>2}m {seconds:2}s"
    elif seconds > 0:
        formatted = f"{seconds:>2}s"
    else:
        formatted = f"{total_seconds:.2f}s"
    return formatted


def get_mask_postfix(mask_classes: Sequence[str]) -> str:
    if mask_classes == ["annotation"]:
        postfix = "_mask-annotation"
    elif mask_classes == ["foreground"]:
        postfix = "_mask-foreground"
    elif set(mask_classes) == set(["foreground", "annotation"]):
        postfix = "_mask-foreground-annotation"
    else:
        postfix = ""
    return postfix


def get_mask_path(path: Path, mask_classes: Sequence[str]) -> Path:
    mask_postfix = get_mask_postfix(mask_classes)
    mask_path = path.with_name(path.stem + mask_postfix + ".png")
    return mask_path


def setup_logging(verbose: bool = False, logfile: Optional[Path] = None):
    log_fmt = logging.Formatter(
        fmt="[%(asctime)s][%(levelname)-1.1s] %(message)s", datefmt="%Y.%m.%d %H:%M:%S"
    )
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(log_fmt)
    logging.root.addHandler(console_handler)
    if verbose:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(logging.INFO)

    if logfile is not None:
        file_handler = logging.FileHandler(filename=str(logfile), mode="a")
        file_handler.setFormatter(log_fmt)
        logging.root.addHandler(file_handler)


def common_path(paths: Sequence[Path]) -> Optional[Path]:
    common_parent = None
    if len(paths) == 0:
        return None
    common_parent = paths[0]
    if len(paths) > 1:
        for path in paths[1:]:
            if common_parent is not None:
                common_parent = common(common_parent, path)

    return common_parent


def common(path_1: Path, path_2: Path) -> Optional[Path]:
    path = None
    for component_1, component_2 in zip(path_1.parts, path_2.parts):
        if component_1 == component_2:
            if path is None:
                path = Path(component_1)
            else:
                path = path.joinpath(component_1)
        else:
            break
    return path


def find_corresponding_path(
    path: Path,
    root: Path,
    candidate_paths: Optional[Sequence[Path]],
    candidate_root: Optional[Path],
    suffix: str,
) -> Optional[Path]:
    if candidate_paths is None:
        return None
    if candidate_root is None:
        return None
    candidate = candidate_root.joinpath(path.relative_to(root)).with_suffix(suffix)
    if candidate in candidate_paths:
        return candidate
    else:
        log.warning(f"Found no corresponding match for {path}")
        return None


def find_files(
    input_paths: Sequence[Path], suffixes: Sequence[str]
) -> Tuple[Optional[List[Path]], Optional[Path]]:
    if input_paths is None:
        file_paths = None
        common_root = None
    if len(input_paths) == 1:
        input_path = input_paths[0]
        file_name = input_path.name
        if input_path.is_file() and any([file_name.endswith(s) for s in suffixes]):
            file_paths = [input_path]
            common_root = input_path.parent
        elif input_path.is_dir():
            file_paths = [p for s in suffixes for p in input_path.rglob(f"*{s}")]
            file_paths = sorted(list(set(file_paths)))
            common_root = input_path
        else:
            log.error(
                f"Input path is not a valid input file nor a folder:\n{input_path}"
            )
            file_paths = None
            common_root = None
    else:
        file_paths = []
        for p in input_paths:
            if p.is_file() and any([p.name.endswith(s) for s in suffixes]):
                file_paths.append(p)
            elif p.is_dir():
                file_paths.extend([f for s in suffixes for f in p.rglob(f"*{s}")])
        if len(file_paths) == 0:
            file_paths = None
            common_root = None
        else:
            file_paths = sorted(list(set(file_paths)))
            common_root = common_path(file_paths)
    return file_paths, common_root


def format_filestem(path: Path) -> str:
    formatted_stem = path.stem
    return formatted_stem


def output_path_from_scan(
    scan_path: Path, scan_root: Path, output_root: Path
) -> Path:
    output_path = output_root.joinpath(scan_path.relative_to(scan_root))
    stem = format_filestem(scan_path)
    output_path = output_path.parent.joinpath(stem + ".png")
    return output_path


def block_from_scan_path(scan_path: Path) -> str:
    block = scan_path.parent.name
    if scan_path.stem.startswith("TCGA"):
        block = scan_path.stem.split(".")[0]
    return block


def scanner_from_path(path: Path) -> Optional[str]:
    scanner = path.parents[1].name
    if scanner.lower() in ["ap", "aperio"]:
        return "aperio"
    elif scanner.lower() in ["xr"]:
        return "xr"
    else:
        log.warning(f"Invalid scanner from {path}: {scanner}")
        return None


def maybe_path(string: Optional[str]) -> Optional[Path]:
    return None if string is None else Path(string)


def extensions_from_scanners(scanners):
    extensions = []
    if scanners is None:
        return None
    for scanner, extension in supported_scanners.items():
        if scanner in scanners:
            extensions.append(extension)
    return extensions


def copy_with_rsync(source: Path, dest: Path, verbose: bool = False):
    cmd = ["rsync", "-av", shlex.quote(str(source)), shlex.quote(str(dest))]
    cmd_str = " ".join(cmd)
    if verbose:
        subprocess.check_call(cmd_str, shell=True)
    else:
        subprocess.check_call(
            cmd_str,
            shell=True,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )


def suffix(count):
    if count == 0 or count > 1:
        suffix = "s"
    else:
        suffix = ""
    return suffix
