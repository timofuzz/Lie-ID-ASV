"""
Shared path helpers for the ASV framework.

We keep the third-party repos side-by-side under the project root:
- PythonVehicleSimulator-master
- LieGroupHamDL-main
- Lie-MPC-AMVs-main (Matlab reference, left untouched here)
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_SRC = REPO_ROOT / "PythonVehicleSimulator-master" / "src"
HAMDL_ROOT = REPO_ROOT / "LieGroupHamDL-main"


def add_vendor_paths() -> None:
    """
    Insert vendor repos into sys.path so imports like
    `python_vehicle_simulator` and `se3hamneuralode` work.
    """
    import sys

    for path in (SIM_SRC, HAMDL_ROOT):
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
