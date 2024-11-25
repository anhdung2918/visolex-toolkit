import unittest
from unittest import TestCase
from visolex import Dictionary

# with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'openai_api_key.txt')) as f:
#     OPENAI_API_KEY = f.readlines()

OPENAI_API_KEY = input('Enter your OpenAI API Key:')

class TestDictionary(TestCase):
    def test_1(self):
        # NSW in dictionary
        dictionary = Dictionary.Instance()
        result = dictionary.search('ko', search_dict=True)
        expected_res = {
            "normalized": ["không"],
            "response": {
                "ko": {
                    "normalized": ["không"],
                    "definition": {"không": "không có, không phải, không đúng, phản đối"},
                    "abbreviations": {"không": ["k", "ko", "k0"]},
                    "example": "Tôi ko hiểu bạn đang nói gì."
                }
            }
        }
        self.assertTrue('ko' in dictionary.nsw_dict.keys())
        self.assertDictEqual(result, expected_res)
        
    def test_2(self):
        # NSW in dictionary -> Not search in dictionary, but using GPT-4 instead
        dictionary = Dictionary.Instance()
        result = dictionary.search('ko', search_dict=False, openai_api_key=OPENAI_API_KEY)
        self.assertTrue(result)
        self.assertGreater(len(result), 0)

    def test_3(self):
        # NSW not in dictionary, search using GPT-4 and add new vocab to dictionary
        dictionary = Dictionary.Instance()
        self.assertFalse('đcs' in dictionary.nsw_dict.keys())
        result = dictionary.search('đcs', add_to_dict=True, openai_api_key=OPENAI_API_KEY)
        self.assertTrue('đcs' in dictionary.nsw_dict.keys())
        self.assertGreater(len(result), 0)

    def test_4(self):
        # NSW not in dictionary, search using GPT-4 and add new vocab to dictionary
        # Save new dictionary to json file
        dictionary = Dictionary.Instance()
        self.assertFalse('csvc' in dictionary.nsw_dict.keys())
        result = dictionary.search('csvc', add_to_dict=True, openai_api_key=OPENAI_API_KEY)
        self.assertTrue('csvc' in dictionary.nsw_dict.keys())
        filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'new_dictionary.json')
        dictionary.save_dict(filepath=filepath)
        self.assertTrue(os.path.exists(filepath))

    def test_5(self):
        # Search using GPT-4 with custom prompt template
        dictionary = Dictionary.Instance()
        custom_prompt = """{nsw} có nghĩa là gì?"""
        result = dictionary.search(
            'csvc', search_dict=False, openai_api_key=OPENAI_API_KEY, prompt_tmpl=custom_prompt
        )
        self.assertGreater(len(result), 0)

if __name__ == '__main__':
    unittest.main()