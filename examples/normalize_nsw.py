from visolex.lexnorm import normalize_sentence

input_str = ''

# Normalize with NSW detection (Return normalized sentence along with NSW detection)
normalize_sentence(input_str)
# Normalize without NSW detection (Only return normalized sentence)
normalize_sentence(input_str, nsw_detection=False)
