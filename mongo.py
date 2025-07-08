# import library for various processes with the OS
import os
# import library for yaml handling
import yaml
# import library for hanlding the MongoDB client
import pymongo
# import library for retrieving datetime
from datetime import datetime, UTC
# import library for hanlding the csv data and transformations
import pandas as pd
import json

from utils import df_rebase


config_path = os.path.join(os.getcwd(), "config.yml")

with open(config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

#print("hi world")

client = pymongo.MongoClient(config["client"])

db = client[config["db"]]

col = db[config["col"]]

#col.insert_one({"test": "ok"})

data_path = os.path.join(os.getcwd(), "data")
print(data_path)

classes_folders_list = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
print(classes_folders_list)

#print files in folder
folder_path = os.path.join(data_path, classes_folders_list[1])
files_in_folder = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
print(files_in_folder)

# root of data dir
DATA_DIR = "data"

#  bypass all subdirectories (class_A, class_B, ...)
for class_name in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_path):
        continue #ignore files

    for file_name in os.listdir(class_path):
        if not file_name.endswith(".csv"):
            continue

        file_path = os.path.join(class_path, file_name)
        df = pd.read_csv(file_path)

        try:
            document = {
                "data": {
                    "acc_x": df["acc_x"].tolist(),
                    "acc_y": df["acc_y"].tolist(),
                    "acc_z": df["acc_z"].tolist(),
                    "gyro_x": df["gyro_x"].tolist(),
                    "gyro_y": df["gyro_y"].tolist(),
                    "gyro_z": df["gyro_z"].tolist(),                   
                },
                "label": class_name,
                "datetime":datetime.now(UTC)
            }

            col.insert_one(document)
            print(f"✅ Inserted {file_name} from {class_name}")
        except KeyError:
            print(f"⚠️ Skipped {file_name}: missing acc_x/y/z - gyro_x/y/z columns")