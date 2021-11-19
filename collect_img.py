# Libraries
import pandas as pd
import numpy as np
import os
from bs4 import BeautifulSoup
import urllib.request
import random
import cv2


# Read the train set
train_set = pd.read_csv("data/train_n11.csv", sep='|')
train_desc = train_set.DESCRIPTION
train_title = train_set.TITLE


# Parse the HTML string
def get_img_url(html_text):
    """
    Find image urls in html string
    """
    soup = BeautifulSoup(html_text, 'html.parser')
    try:
        imgs = soup.find_all("img")
        urls = [img["src"] for img in imgs if len(img["src"]) > 0]
    except:
        urls = []
    return urls


# Create files to save images
for set_split in ["train", "test"]:
    for cat_id in np.unique(train_set["CATEGORY_ID"]):
        newpath = r'data/images_224x224/{}/{}'.format(set_split, str(cat_id))
        if not os.path.exists(newpath):
            os.makedirs(newpath)


# Save images
urls = train_desc.map(lambda x: get_img_url(x))
urls_df = pd.DataFrame(urls)
urls_df.rename(columns={"DESCRIPTION": "urls"}, inplace=True)
urls_df["cat_id"] = train_set["CATEGORY_ID"]


def save_img(url, img_size):
    train_test_decision = "train" if random.random() < 0.8 else "test"
    try:
        last_number_in_file = int(max([filename[:4] for filename in sorted(os.listdir("data/images_224x224/{}/{}".format(train_test_decision, cat_id)))]))
    except:
        last_number_in_file = 0
    filename = "{0:0=4d}".format(last_number_in_file + 1)
    path = "data/images_224x224/{}/{}/{}.jpg".format(train_test_decision, cat_id, filename)
    try:
        urllib.request.urlretrieve(url, path)
        img = cv2.imread(path)
        img = cv2.resize(img, (img_size, img_size))
        cv2.imwrite(path, img)
    except:
        pass
    

categories = list()
for row in range(len(urls_df)):
    [print(row) if row % 1000 == 0 else ""]
    cat_id = urls_df["cat_id"][row]
    [categories.append(cat_id) for _ in range(len(urls_df["urls"][row]))]
    if categories.count(cat_id) < 3000 and len(urls_df["urls"][row]) > 0:
        pd.Series(urls_df["urls"][row]).map(lambda x: save_img(url=x, img_size=224))
