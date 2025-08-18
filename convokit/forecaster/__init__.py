from .forecaster import *
from .forecasterModel import *
from .cumulativeBoW import *
import sys

# Import CRAFT models if torch is available
if "torch" in sys.modules:
    from .CRAFTModel import *
    from .CRAFT import *

# Import Transformer models with proper error handling
try:
    from .TransformerDecoderModel import *
except ImportError as e:
    if "Unsloth GPU requirement not met" in str(e):
        print(
            "Error from Unsloth: NotImplementedError: Unsloth currently only works on NVIDIA GPUs and Intel GPUs."
        )
    elif "Unsloth is not currently available on macOS" in str(e):
        print(
            "TransformerDecoderModel: If you are a mac user, unsloth is currently not available on macOS. For other users, please use 'pip install convokit[llm]' to install LLM related dependencies."
        )
    elif "not currently installed" in str(e):
        print(
            "TransformerDecoderModel: ML dependencies are not currently installed. Run 'pip install convokit[llm]' to install them (or 'pip install convokit[llmmac]' for macOS users)."
        )
    else:
        raise

try:
    from .TransformerEncoderModel import *
except ImportError as e:
    if "not currently installed" in str(e):
        print(
            "TransformerEncoderModel: ML dependencies are not currently installed. Run 'pip install convokit[llm]' to install them (or 'pip install convokit[llmmac]' for macOS users)."
        )
    else:
        raise

from .TransformerForecasterConfig import *
