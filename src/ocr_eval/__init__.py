"""Evaluation harness for the Arabic document OCR fine-tune.

The harness never loads the model. It scores predictions that already exist on
disk, which is what lets it run on an ordinary CPU workstation with no cluster
access, and what lets it score output from any other OCR system used as a
comparator (research R-012).
"""

__version__ = "0.2.0"

# Re-exported so `ocr-eval --version` and every run manifest read the policy
# version from one place. A change to the normalisation tables bumps this and
# makes old reports formally incomparable (research R-004).
from ocr_eval.normalise.tables import POLICY_VERSION

__all__ = ["__version__", "POLICY_VERSION"]
