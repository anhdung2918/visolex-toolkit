from visolex.lexnorm import detect_nsw
import argparse

if __name__ == "__main__":
    # Arguments Setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_str", help="Input string", type=str)
    args = parser.parse_args()

    # Detect NSW
    nsw_spans = detect_nsw(args.input_str)
    for i in range(len(nsw_spans)):
        print(f"{i + 1}. NSW '{nsw_spans[i]['nsw']}' start from index {nsw_spans[i]['start_index']} to index {nsw_spans[i]['end_index']}")