from visolex.dictionary import Dictionary
import argparse

if __name__ == "__main__":
    # Arguments Setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsw", help="NSW", type=str)
    parser.add_argument("--search_dict", action="store_true", help="Whether to search in predefined dictionary")
    parser.add_argument("--add_to_dict", action="store_true", help="Whether to add non-existing NSW to the predefined dictionary")
    parser.add_argument("--openai_api_key", help="Your OpenAI API key", type=str)
    args = parser.parse_args()

    # Initialize a dictionary
    dictionary = Dictionary()
    # Search NSW
    response_str = dictionary.search(args.nsw, args.search_dict, args.add_to_dict, args.openai_api_key)
    print(f"Response:\n{response_str}")
