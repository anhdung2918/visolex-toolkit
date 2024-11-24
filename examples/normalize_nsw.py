from visolex.lexnorm import normalize_sentence
import argparse

if __name__ == "__main__":
    # Arguments Setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_str", help="Input string", type=str)
    parser.add_argument("--nsw_detection", action="store_true", help="Whether to detect NSW")
    args = parser.parse_args()

    # Normalize
    if args.nsw_detection:
        nsw_spans, pred_str = normalize_sentence(args.input_str, args.nsw_detection)
        print(f"Normalized sentence: {pred_str}")
        print(f"NSW Detection:\n")
        for i in range(len(nsw_spans)):
            print(f"{i + 1}. NSW '{nsw_spans[i]['nsw']}' ==> STANDARD FORM: '{nsw_spans[i]['prediction']}' (CONFIDENCE : {nsw_spans[i]['confidence_score']})")
    else:
        pred_str = normalize_sentence(args.input_str, args.nsw_detection)
        print(f"Normalized sentence: {pred_str}")
