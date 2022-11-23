import pandas as pd
import json
import argparse


def main():
    # Read in arguments from Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle-file', type=str, default="../data/multi_label_train_v0.3.pkl",
                        help='Path to input pickle file')
    parser.add_argument('--output-file', type=str, default="../data/multi_label_train_v0.x.json",
                        help='Path to output json file')
    parser.add_argument('--num-samples', type=int, default=-1, help='Number of samples to take')
    parser.add_argument('--ignore-unknown-classes', default=True,
                        help='Include unknown classes', action='store_false')
    args = parser.parse_args()

    df = pd.read_pickle(args.pickle_file)
    df = df[df["valid_image"] == True]

    if args.num_samples > 0:
        df = df.sample(args.num_samples, random_state=args.num_samples)
    
    if not args.ignore_unknown_classes:
        df = df[(df["incidents_list"] != "unknown") & df["places_list"] != "unknown"]

    # Save df to json with line formatting
    values_dict = df[["incidents", "places"]].to_dict(orient="records")
    final_dict = {k: v for k, v in zip(df["key"], values_dict)}

    # Save final_dict to json
    with open(args.output_file, "w") as f:
        json.dump(final_dict, f)


if __name__ == "__main__":
    main()
