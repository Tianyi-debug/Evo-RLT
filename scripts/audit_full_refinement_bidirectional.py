#!/usr/bin/env python3
"""Repository-local entrypoint; no package reinstall is required."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evo_rlt.cli.audit_full_refinement_bidirectional import main


if __name__ == "__main__":
    main()
