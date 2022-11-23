import pandas as pd
import json
import os

def rename_images(df: pd.DataFrame, path_to_images: str):
    df = df[df["downloadable"] == True]

    # For every row in the dataframe, rename the image
    for index, row in df.iterrows():
        old_name = path_to_images + row["image_id"] + ".jpg"
        new_name = path_to_images + row["key"]
        #os.rename(old_name, new_name)

        if index % 10000 == 0:
            print("Renamed {} images".format(index))
            print("Old name: {}".format(old_name))
            print("New name: {}".format(new_name))

def main():

    path_to_images = "../data/images/"
    path_to_df = "../data/multi_label_train_v0.4.pkl"

    df = pd.read_pickle(path_to_df)
    rename_images(df, path_to_images)

    path_to_df = "../data/multi_label_val_v0.4.pkl"
    df = pd.read_pickle(path_to_df)
    rename_images(df, path_to_images)


if __name__ == "__main__":
    main()