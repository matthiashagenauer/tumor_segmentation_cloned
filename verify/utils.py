import numpy as np


class Formats:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_test_result(name, ok):
    dots = "." * max(80 - len(name), 0)
    if ok:
        print(f"{name}{dots}: {Formats.OKGREEN}OK{Formats.ENDC}")
    else:
        print(f"{name}{dots}: {Formats.FAIL}FAIL{Formats.ENDC}")


def start_end_coord(part_name, prefix):
    """
    Expected input

    <prefix>-<start>-<end>
    """
    assert part_name.startswith(prefix), f"Part must start with {prefix}"
    end_part_name = part_name.replace(prefix + "-", "")
    split = end_part_name.split("-")
    assert len(split) == 2, f"Incorrect format in '{part_name}'"
    return int(split[0]), int(split[1])


def coords_from_tile_name(tile_name):
    """
    Expected input is a file name without suffix on the form

    <prefix>_rows-<start row>-<end row>_cols-<start col>-<end col>
    """
    assert "." not in tile_name
    name_split = tile_name.split("_")
    row_part = name_split[-2]
    col_part = name_split[-1]
    try:
        row_start, row_end = start_end_coord(row_part, "rows")
        col_start, col_end = start_end_coord(col_part, "cols")
    except AssertionError as e:
        print(f"Exception in tile {tile_name}: {e}")
        exit()
    return row_start, row_end, col_start, col_end


def coords_from_tile_folder(tile_folder, extension):
    names = [p.stem for p in tile_folder.glob(f"*{extension}")]
    coords = [coords_from_tile_name(n) for n in names]
    return coords


def find_dimensions_from_tiles(tile_folder, extension):
    coords = coords_from_tile_folder(tile_folder, extension)
    if len(coords) == 0:
        return None, None
    row_starts = [coord[0] for coord in coords]
    row_ends = [coord[1] for coord in coords]
    col_starts = [coord[2] for coord in coords]
    col_ends = [coord[3] for coord in coords]
    assert min(row_starts) == 0, f"Row starts at {min(row_starts)} in {tile_folder}"
    assert min(col_starts) == 0, f"Column starts at {min(row_starts)} in {tile_folder}"
    height = max(row_ends)
    width = max(col_ends)
    return height, width


def normalise_scanner(name):
    name = name.lower()
    if name in ["ap", "aperio"]:
        return "aperio"
    elif name in ["xr"]:
        return "xr"
    elif name in ["mrxs", "mirax"]:
        return "mirax"
    elif name in ["s210"]:
        return "s210"
    elif name in ["scanner"]:
        return "scanner"
    else:
        print(f"ERROR: Unexpected scanner: '{name}'")
        exit()


def is_mesh_complete(coords):
    """
    Coords: list of tuples with tile coordinates. start coords are inclusive, end coords
    are exclusive.

    [(start_row, end_row, start_col, end_col), ...]
    """
    rows = np.unique([coord[0] for coord in coords])
    cols = {}
    ok = True
    for row in rows:
        candidate = sorted([coord[2] for coord in coords if coord[0] == row])
        if row in cols.keys():
            if candidate != cols[row]:
                print(f"WARNING: Invalid cols for row {row}: {candidate}")
                ok = False
            else:
                cols[row] = candidate
    return ok


def are_tiles_present(tile_folder, extension):
    if not tile_folder.exists():
        print(f"WARNING: Tile folder does not exis: {tile_folder}")
        return False
    coords = coords_from_tile_folder(tile_folder, extension)
    return is_mesh_complete(coords)
