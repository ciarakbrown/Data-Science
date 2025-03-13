import os
import requests
from zipfile import ZipFile
from urllib.parse import urlparse

# make the directory if not exist
os.makedirs("data/raw", exist_ok=True)

# url for downloading the dataset
urls = [
    "https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip",
    "https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip"
]

for url in urls:
    # unzip the dataset into the directory same as zip file name
    zip_name = os.path.basename(urlparse(url).path)
    zip_path = os.path.join("data", zip_name)

    # download dataset
    with requests.get(url) as r:
        with open(zip_path, "wb") as f:
            f.write(r.content)

    # unzip the data
    with ZipFile(zip_path) as z:
        z.extractall("data/raw")

    # delete the zip file after unzipped
    os.remove(zip_path)

print("data has been downloaded successfully")