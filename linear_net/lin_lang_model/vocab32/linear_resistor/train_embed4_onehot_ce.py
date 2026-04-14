#!/usr/bin/env python3
"""Wrapper entry point for the vocab32 linear-resistor trainer."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().parents[1] / "clln_lang_trainer_embed4_onehot_ce_linear_resistor.py"
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
