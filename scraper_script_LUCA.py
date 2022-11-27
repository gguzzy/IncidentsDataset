import requests
from PIL import Image
import json
import os
import configargparse

# Custom parser
def get_parser():
    parser = configargparse.ArgumentParser(description="Incident Model scraper program.")
    parser.add_argument("--path_folder",
                        default="IncidentsDataset/data",
                        type=str,
                        help="Insert the absolut path of the folder containing the json file containing the images info.")
    parser.add_argument("--name_file",
                        default="multi_label_train.json",
                        type=str,
                        help="Insert the name of the json file containing the images info.")
    parser.add_argument("--images_folder",
                        default="images_eccv_train",
                        type=str,
                        help="The images will be created in the 'path folder'.")
    parser.add_argument("--n_images",
                        default=10,
                        type=int,
                        help="How many images you want to consider.")
    # Work in progress.
    parser.add_argument("--filter_images",
                        default=False,
                        type=bool,
                        help="The option to filter out the categories.")
    parser.add_argument("--filter_incidents",
                        default=["flooded","traffic jam"],
                        type=set,
                        help="Categories to filter (places).")
    parser.add_argument("--filter_places",
                        default=["river"],
                        type=set,
                        help="Categories to filter (places).")              
    return parser

def download_images(args):
  string_path = os.path.join(args.path_folder, args.name_file)
  new_string_path = os.path.join(args.path_folder, "new_"+args.name_file)

  with open(string_path, "r") as fp:
    dataset = json.load(fp)
  new_dataset = {}
  cont = 0
  cont_error = 0
  for image_name in dataset.keys():
    ex = False
    cont += 1
    if (args.filter_images == True):
      incidents, places = set(), set()
      for i in args.filter_incidents:
        incidents.add(i)
      for i in args.filter_places:
        places.add(i)
      # Save the length
      len_i = len(incidents)
      len_p = len(places)
      # For every item added look if the set len is not changing (set can't have duplicate)
      for i in dataset[image_name]["incidents"].keys():
        if (dataset[image_name]["incidents"][i] == 1):
          incidents.add(i)
          if( len(incidents) == len_i ):
            ex = True
            break
          else:
            len_i = len(incidents)
      for i in dataset[image_name]["places"].keys():
        if (dataset[image_name]["places"][i] == 1):
          places.add(i)
          if( len(places) == len_p ):
            ex = True
            break
          else:
            len_p = len(places)
        # filter our categories
      # Incident or place prohibited NOT found
      if ex == False:
        img_url = dataset[image_name]["url"]
        try:
          img = Image.open(requests.get(img_url, stream=True, timeout=3).raw)
          img.save(os.path.join(args.path_folder, args.images_folder, image_name.split("/")[0] + "_" + image_name.split("/")[1]))
          new_dataset[image_name.split("/")[0] + "_" + image_name.split("/")[1]] = dataset[image_name]

        except Exception as e:
          cont_error += 1
          print(str(e))
      else:
        print(f"Excluded image: {dataset[image_name]}")
    else:
      img_url = dataset[image_name]["url"]
      try:
        img = Image.open(requests.get(img_url, stream=True, timeout=3).raw)
        img.save(os.path.join(args.path_folder, args.images_folder, image_name.split("/")[0] + "_" + image_name.split("/")[1]))
        new_dataset[image_name.split("/")[0] + "_" + image_name.split("/")[1]] = dataset[image_name]

      except Exception as e:
        cont_error += 1
        print(str(e))

    if cont == args.n_images:
      break
  print(f"Debug: Broken link found while loading images: {cont_error} - {round((cont_error / args.n_images) * 100, 2)}%")
  # Save the clean json
  try:
    with open(new_string_path, 'w') as fp:
      json.dump(new_dataset, fp)
      print(f">>> Saved {new_string_path} correctly <<<")
  except Exception as e:
      print(str(e))
  return


# Work in progress.
def scout_images(args):
  string_path = os.path.join(args.path_folder, args.name_file)
  with open(string_path, "r") as fp:
    dataset = json.load(fp)
  cont = 0
  cont_error = 0
  cont_neg = 0
  for image_name in dataset.keys():
    cont += 1
    img_url = dataset[image_name]["url"]
    try:
        img = Image.open(requests.get(img_url, stream=True, timeout=3).raw)
        if (sum(dataset[image_name]["incidents"].values()) == 0):
          cont_neg += 1
          # Temporary fix..
          img.save(os.path.join(args.path_folder, args.images_folder, image_name.split("/")[0] + "_" + image_name.split("/")[1]))
    except Exception as e:
        cont_error += 1
        print(f"Numero di link rotti: {cont_error}")
    if cont == args.n_images:
        break
  print(f"Only negative images --> Debug: Broken link found while loading images: {round((cont_neg / args.n_images-cont_error) * 100, 2)}%")
  

if __name__ == '__main__':
  args = get_parser().parse_args()
  download_images(args)

# Need to target different images (only_pos, pos_and_neg e classes for test and val)
















