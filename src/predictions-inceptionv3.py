# Libraries
import numpy as np
import pandas as pd
import cv2
from bs4 import BeautifulSoup
import urllib.request
!pip install pickle5
import pickle5 as pickle
from keras.models import load_model


# Load the model
model = load_model('inceptionV3_model.h5')


# Load the dictionary that contains labels
def load_labels_dict():
    with open("labels.pkl", 'rb') as f:
        return pickle.load(f)
labels_dict = load_labels_dict()
labels_dict = labels_dict[0]


# Read the test data
test_set = pd.read_csv("data/test_n11.csv", sep='|')
test_desc = test_set.DESCRIPTION


def parse_and_predict_img(html_text, img_size, labels):
    """
    Find image urls in html string and predict using the model
    """
    predictions = []
    soup = BeautifulSoup(html_text, 'html.parser')
    try:
        imgs = soup.find_all("img")
        urls = [img["src"] for img in imgs if len(img["src"]) > 0]
        count_img = 0
        for img in urls:
            if count_img < 11:
                request_img = urllib.request.urlretrieve(img, "img.jpg")
                img = cv2.imread('img.jpg')
                img = cv2.resize(img,(img_size, img_size))
                img = img.reshape(1, img_size,img_size, 3)
                prediction_index = np.argmax(model.predict(img))
                prediction_class = int(list(labels.keys())[list(labels.values()).index(prediction_index)])
                predictions.append(prediction_class)
                count_img += 1
                
    except:
        predictions = []
    return predictions


# Predictions
predictions = test_desc.map(lambda x: parse_and_predict_img(x, img_size=224, labels=labels_dict))
predictions.to_csv("data/inception_predictions.csv")
