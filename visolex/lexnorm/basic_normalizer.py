import re
from typing import List, Dict, Any, Union
from visolex.framework_components.regex_expression import Protected, emoji_pattern, emoji_list, tone_dict_map
from visolex.framework_components.preprocessing import tone_normalization, split_emoji_text, split_emoji_emoji, simple_tokenize

class BasicNormalizer:
    def __init__(args, logger):
        self.args = args
        self.logger = self.logger

    # Preprocessing pipeline
    def basic_preprocess(self, text: str,  lowercase: bool = False):
        if lowercase:
            text = text.lower()
        text = tone_normalization(text, tone_dict_map)
        text = split_emoji_text(text, emoji_list)
        tokens = simple_tokenize(text, [Protected], emoji_pattern)
        tokens = split_emoji_emoji(tokens, emoji_list)
        return ' '.join(filter(str.strip, tokens))
