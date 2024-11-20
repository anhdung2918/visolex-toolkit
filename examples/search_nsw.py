from visolex.dictionary import Dictionary

# Initialize a dictionary
dictionary = Dictionary()

# Search a word available in dictionary
dictionary.search(nsw = 'ko')
# Search a word that not in dictionary and add new word to dictionary
print("DICTIONARY SIZE BEFORE SEARCHING", len(dictionary.nsw_dict))
dictionary.search(nsw = 'cđm', add_to_dict=True, openai_api_key="<your-openai-api-key>")
print("DICTIONARY SIZE AFTER SEARCHING", len(dictionary.nsw_dict))
# Search with chatgpt
dictionary.search_chatgpt(nsw = 'cđm', openai_api_key="<your-openai-api-key>")

# search(nsw, search_dict=True, add_to_dict=False, openai_api_key=None, prompt_tmpl=None)
# search_chatgpt(nsw, openai_api_key, prompt_tmpl, add_to_dict