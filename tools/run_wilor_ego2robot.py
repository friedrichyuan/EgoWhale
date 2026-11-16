#!/usr/bin/env python3
"""Run only EgoWhale's ego2robot-synchronized WiLoR node."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egowhale.wilor.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
