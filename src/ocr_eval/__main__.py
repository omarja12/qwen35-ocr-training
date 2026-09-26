"""Entry point for `python -m ocr_eval`.

Two invocation paths exist and they must run the same code. On a workstation
the console script `ocr-eval` is available after `pip install -e .`. Inside the
training pod the package is not pip-installed at all — the Containerfile copies
`src/` to /opt/ocr-training/src — so the invocation there is:

    PYTHONPATH=/opt/ocr-training/src python -m ocr_eval

This module exists so that difference is a path difference and nothing more.
"""

import sys

from ocr_eval.cli import main

if __name__ == "__main__":
    sys.exit(main())
