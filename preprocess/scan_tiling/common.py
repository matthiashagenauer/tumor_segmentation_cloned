import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2].joinpath("common")))

import common_utils  # type: ignore
import scan_utils  # type: ignore
