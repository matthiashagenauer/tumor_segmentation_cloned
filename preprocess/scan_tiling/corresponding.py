import sys
import logging
from pathlib import Path
from typing import Optional, Sequence

from common import common_utils

log = logging.getLogger()


class Case:
    def __init__(
        self,
        scan_path: Path,
        annotation_path: Optional[Path],
        mask_path: Optional[Path],
        scan_output_path: Path,
        annotation_output_path: Optional[Path],
        relative_dir: Path,
    ):
        # Path to scan
        self.scan = scan_path
        # Optional path to annotation
        self.annotation = annotation_path
        # Optional path to mask
        self.mask = mask_path
        # Path to parent folder of resulting tiles for this scan
        self.scan_output = scan_output_path
        # Path to parent folder of resulting tiles for this annotation
        self.annotation_output = annotation_output_path
        # Path between scan root and scan (exclusive scan filename)
        self.relative_dir = relative_dir
        # Flag to signal if this case can be removed from the cache
        self.remove_scan = False

    def __repr__(self) -> str:
        annotation = "None" if self.annotation is None else self.annotation
        annotation_output = (
            "None" if self.annotation_output is None else self.annotation_output
        )
        mask = "None" if self.mask is None else self.mask
        string = f"Scan       input:  {self.scan}\n"
        string += f"           output: {self.scan_output}\n"
        string += f"Annotation input:  {annotation}\n"
        string += f"           output: {annotation_output}\n"
        string += f"Mask       input:  {mask}"
        return string


def find_corresponding_paths(
    scan_paths: Sequence[Path],
    scan_root: Path,
    annotation_paths: Optional[Sequence[Path]],
    annotation_root: Optional[Path],
    mask_root: Optional[Path],
    mask_appendix: Optional[str],
    output_root: Path,
    add_scan_folder: bool,
):
    cases = []
    for scan in sorted(scan_paths):

        scan_output = common_utils.output_path_from_scan(
            scan, scan_root, output_root
        ).with_suffix("")
        if add_scan_folder:
            scan_output = scan_output.parent.joinpath(
                scan_output.stem, scan_output.name
            )
        relative_dir = scan.relative_to(scan_root).parent
        if annotation_root is None:
            annotation = None
            annotation_output = None
        else:
            annotation_output = scan_output.with_name(
                scan_output.name + "_mask-annotation"
            )
            if not annotation.exists():
                log.error(f"Could not find annotation path corresponding to: {scan}")
                log.error(f"Tried to locate {annotation}")
                sys.exit()
            if annotation_output == scan_output:
                log.error(f"Equal scan and annotation output: {annotation_output}")
                sys.exit()
        mask = None
        if mask_root is not None:
            mask_root = Path(mask_root)
            mask = mask_root.joinpath(
                relative_dir, scan_output.stem + mask_appendix + ".png"
            )
            if not mask.exists():
                mask = None
        cases.append(
            Case(scan, annotation, mask, scan_output, annotation_output, relative_dir)
        )
    return cases
