from visolex import basic_normalizer
import argparse

if __name__ == "__main__":
    # Arguments Setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_str", help="Input string", type=str)
    parser.add_argument("--lowercase", action="store_true", help="Convert text to lowercase.")
    args = parser.parse_args()

    # Normalize
    processed_text = basic_normalizer(
        text=args.input_str, 
        lowercase=args.lowercase
    )

    print("Processed Text:", processed_text)