"""zhjudge: a Chinese-first, open typed-decision ("System One") model in the Laya checkpoint format."""

import os as _os

# mmBERT ships only pytorch_model.bin. Loading it makes transformers start a NON-daemon thread that asks the Hub's
# safetensors-conversion bot to open a PR on the base model repo; the process then cannot exit until that finishes.
_os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "0.1.0"
