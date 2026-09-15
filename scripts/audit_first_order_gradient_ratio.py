#!/usr/bin/env python3
"""Repository-local first-order Q/BC gradient-ratio entrypoint."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evo_rlt.cli.audit_first_order_gradient_ratio import main


if __name__ == "__main__":
    main()
