import warnings

try:
    from .model import *
    from .util import *
    from .coordination import *
    from .politenessStrategies import *
    from .transformer import *
    from .convokitPipeline import *
    from .hyperconvo import *
    from .speakerConvoDiversity import *
    from .text_processing import *
    from .phrasing_motifs import *
    from .prompt_types import *
    from .classifier.classifier import *
    from .ranker import *
    from .forecaster import *
    from .fighting_words import *
    from .paired_prediction import *
    from .bag_of_words import *
    from .expected_context_framework import *
    from .surprise import *
    from .convokitConfig import *
    from .redirection import *
    from .pivotal_framework import *
    from .utterance_simulator import *
except ModuleNotFoundError as e:
    # Don't print ModuleNotFoundError messages as they're handled by individual modules
    if "not currently installed" not in str(e):
        print(f"An error occurred: {e}")
        warnings.warn(
            "If you are using ConvoKit with Google Colab, incorrect versions of some packages (ex. scipy) may be imported while runtime start. To fix the issue, restart the session and run all codes again. Thank you!"
        )
except Exception as e:
    print(f"An error occurred: {e}")
    warnings.warn(
        "If you are using ConvoKit with Google Colab, incorrect versions of some packages (ex. scipy) may be imported while runtime start. To fix the issue, restart the session and run all codes again. Thank you!"
    )


# __path__ = __import__('pkgutil').extend_path(__path__, __name__)
