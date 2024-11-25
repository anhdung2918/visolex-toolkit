"""
ViSoLex
"""

import os
import sys
from functools import lru_cache

__author__ = """Ha Dung Nguyen"""
__email__ = 'dungngh@uit.edu.vn'

# Check python version
try:
    version_info = sys.version_info
    if version_info < (3, 10, 0):
        raise RuntimeError("ViSoLex requires Python 3.y or later")
except Exception:
    pass

###########################################################
# METADATA
###########################################################

# Version
try:
    version_file = os.path.join(os.path.dirname(__file__), 'VERSION')
    with open(version_file, 'r') as infile:
        __version__ = infile.read().strip()
except NameError:
    __version__ = 'unknown (running code interactively?)'
except IOError as ex:
    __version__ = "unknown (%s)" % ex

###########################################################
# TOP-LEVEL MODULES
###########################################################
from .framework_components.trainer import ViSoLexTrainer
from .lexnorm import detect_nsw, normalize_sentence, basic_normalizer
from .dictionary import Dictionary

optional_imports = {
    'NswDetector': 'visolex.lexnorm.detect',
    'ViSoLexNormalizer': 'visolex.lexnorm.normalize',
    'BasicNormalizer': 'visolex.lexnorm.basic_normalizer'
}

@lru_cache(maxsize=None)
def get_optional_import(module_name, object_name):
    """
    Dynamic Import with @lru_cache:
    - Tries to import a module and retrieve the desired object
    - Returns None if the module isn't available, avoiding import errors.
    """
    try:
        module = __import__(module_name, fromlist=[object_name])
        return getattr(module, object_name)
    except ImportError:
        return None

# Each feature is added to the global namespace if successfully loaded.
for name, module in optional_imports.items():
    globals()[name] = get_optional_import(module, name)

# Explicitly declares which attributes and functions are considered part of the library's public API.
__all__ = [
    'detect_nsw',
    'normalize_sentence',
    'basic_normalizer',
    'ViSoLexTrainer',
    'Dictionary',
    'NswDetector',
    'ViSoLexNormalizer',
    'BasicNormalizer'
]