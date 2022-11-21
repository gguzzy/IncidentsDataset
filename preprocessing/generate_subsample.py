import pandas as pd
import json
import argparse


def main():
    # Read in arguments from Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle-file', type=str, default="../data/multi_label_train_v0.3.pkl",
                        help='Path to input pickle file')
    parser.add_argument('--output-file', type=str, defaults="../data/multi_label_train_v0.x.json", help='Path to output json file')
    args = parser.parse_args()

    df = pd.read_pickle(args.pickle_file)
    df = df[df["downloadable"] == True]

    # Save df to json with line formatting
    values_dict = df.to_dict(orient="records")

    final_dict = {k: v for k, v in zip(df["image_id"], values_dict)}

    # Save final_dict to json
    with open(args.output_file, "w") as f:
        json.dump(final_dict, f, indent=4)


if __name__ == "__main__":
    main()
