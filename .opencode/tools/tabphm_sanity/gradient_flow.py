"""Standalone wrapper for the PICID gradient-flow sanity check."""

from __future__ import annotations

import json
import sys

from PICID_sanity import RESULT_MARKER, run_command
from PICID_sanity_core import json_default


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python .opencode/tools/PICID_sanity/gradient_flow.py '<json-payload>'"
        )
    result = run_command("gradient_flow", json.loads(sys.argv[1]))
    print(RESULT_MARKER + json.dumps(result, default=json_default, sort_keys=True))
