"""Building blocks for the RAG optimization course."""

import os
from importlib.metadata import version

# DeepEval sends usage statistics unless told otherwise: keep course runs private.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

# Single source of truth: the version declared in pyproject.toml.
__version__ = version("ragcourse")
