# Libraries
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')


# Read the train file
train_set = pd.read_csv("data/train_n11.csv", sep='|')


# Read the test file
test_set = pd.read_csv("data/test_n11.csv", sep='|')


# Create the necessary functions
def find_text(text):
    """
    Find text in HTML string
    """
    text = BeautifulSoup(text, 'html.parser').text
    text = text.replace('\n', '')
    text = re.sub(' +', ' ', text)
    text = text.strip()
    return text


def text_preprocessing(data, lower=True, tokenize=True,
                       isalnum=True, stpwrds=True):
    """
    Preprocessing of the text data
    """
    if lower:
        data = data.map(lambda x: x.lower())
    if tokenize:
        data = data.map(lambda x: nltk.word_tokenize(x))  # Tokenization
    if isalnum:
        data = data.map(lambda x: [word for word in x if word.isalnum()])  # Getting only the alphabet letters (a-z) and numbers (0-9) 
    if stpwrds:
        stop_words = stopwords.words('turkish')  
        data = data.map(lambda x: [word for word in x if not word in stop_words])  # Removing stop words
    return data


# Preprocessing of train set
train_set["text"] = train_set["TITLE"] + " " + train_set["DESCRIPTION"]
train_text = train_set["text"].map(lambda x: find_text(x))
train_text_processed = text_preprocessing(train_text)
train_text_joined = train_text_processed.map(lambda x: " ".join(x))
train_set["fasttext_text"] = "__label__" + train_set["CATEGORY_ID"].astype(str) + " " + train_text_joined


# Preprocessing of test set
test_set["text"] = test_set["TITLE"] + " " + test_set["DESCRIPTION"]
test_text = test_set["text"].map(lambda x: find_text(x))
test_text_processed = text_preprocessing(test_text)
test_text_joined = test_text_processed.map(lambda x: " ".join(x))


# Save the files
train_set.to_csv("data/train_preprocessed.csv")
test_text_joined.to_csv("data/test_preprocessed.csv")