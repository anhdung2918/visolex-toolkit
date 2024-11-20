import json
from bs4 import BeautifulSoup
from visolex.utils import Singleton
from visolex.global_variables import DICT_PATH
from visolex.llm.gpt import run_chatgpt
from visolex.llm.prompts import DEFAULT_NSW_SEARCH_PROMPT

def parse_html_to_string(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
    cleaned_text = "\n".join([line for line in text.splitlines() if line.strip()])
    return cleaned_text

@Singleton
class Dictionary:
    def __init__(self, filepath=None):
        """ contains 13k words in this version
        """
        if filepath is None:
            filepath = DICT_PATH
        with open(filepath, 'r', encoding='utf-8') as f:
            self.nsw_dict = json.load(f)

    def add_vocab(self, nsw, response):
        self.nsw_dict[nsw] = {}
        self.nsw_dict[nsw]['response'] = response

    def search_dict(self, nsw):
        normalized_data = self.nsw_dict[nsw]
        result = normalized_data
        try:
            normalized_data = result['response']['normalized'][0]
            response = {
                'word': normalized_data['word'],
                'definition': normalized_data['definition'],
                'abbreviation': normalized_data.get('abbreviations', ''),
                'example': normalized_data.get('example', '')
            }
            response_str = ""
            for key, value in response.items():
                string = "- " + key.upper() + ": " + value + "\n"
                response_str += string
        except:
            response_str = json.dumps(result)
        return response_str

    def search_chatgpt(self, nsw, openai_api_key, prompt_tmpl, add_to_dict):
        prompt = DEFAULT_NSW_SEARCH_PROMPT.format(nsw) if prompt_tmpl is None else prompt_tmpl.format(nsw)
        response = run_chatgpt(nsw, openai_api_key, prompt)
        response = response.replace("```html\n", "").replace("\n```", "")
        if add_to_dict:
            self.add_vocab(nsw, response)
        if prompt is None:
            response_str = parse_html_to_string(response)
        else:
            response_str = response
        return response_str

    def search(self, nsw, search_dict=True, add_to_dict=False, openai_api_key=None, prompt_tmpl=None):
        if search_dict:
            if nsw in self.nsw_dict:
                response_str = self.search_dict(nsw)
            else:
                response_str = self.search_chatgpt(nsw, openai_api_key, prompt_tmpl, add_to_dict)
        else:
            response_str = self.search_chatgpt(nsw, openai_api_key, prompt_tmpl, add_to_dict)
        
        return response_str