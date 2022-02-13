# Libraries
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
import gensim
import fasttext
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')


# Read the train file
train_set = pd.read_csv("data/train_n11.csv", sep='|')


# Read the test file
test_set = pd.read_csv("data/test_n11.csv", sep='|')


# Read the submission file
submission = pd.read_csv("data/sample_submission_n11.csv")


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


def get_img_url(html_text):
    """
    Find image urls in HTML string
    """
    soup = BeautifulSoup(html_text, 'html.parser')
    try:
        imgs = soup.find_all("img")
        urls = [img["src"] for img in imgs if len(img["src"]) > 0]
    except:
        urls = []
    return urls


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


def write_to_txt(filename, text):
    """
    Create a txt file
    """
    with open(filename, 'w', encoding="utf-8") as f:
        f.write(text)


def fasttext_predict(model, row):
    """
    Predict category ids using fasttext
    """
    label = model.predict(row)[0]
    return int(label[0].split("__")[-1])


def most_common(lst):
    """
    Find most frequent element in a given list
    """
    return max(set(lst), key=lst.count)


def image_prediction(row):
    """
    If the image number is greater than 5, predict most common category
    """
    frequency = {}
    for item in row:
        if item in frequency:
            frequency[item] += 1
        else:
            frequency[item] = 1
    if len(eval(row)) > 4:
        if eval(row).count(most_common(eval(row))) > len(eval(row)) / 2:
            prediction = most_common(eval(row))
        else:
            prediction = 0
    else:
        prediction = 0
    return(prediction)


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


# Exploratory data analysis
train_set.head(5)


train_set.info()


test_set.info()


train_set.isnull().sum()


test_set.isnull().sum()


# Distribution of number of words
nr_words_train = train_text_processed.apply(lambda x: len(x))
nr_words_test = test_text_processed.apply(lambda x: len(x))


fig, ax = plt.subplots(1,2, figsize=(15, 6))
sns.histplot(nr_words_train, ax=ax[0],  color='skyblue', stat='density')
sns.kdeplot(nr_words_train, ax=ax[0], color='navy', fill=False, lw=3)
ax[0].set_title('Distribution of Number of Words in Train Set', fontsize=20)
ax[0].set_xlabel('Word Number', fontsize=15)
ax[0].set_ylabel('Density', fontsize=15)
ax[0].tick_params(labelsize=12)


sns.histplot(nr_words_test, color='lightsalmon', stat='density')
sns.kdeplot(nr_words_test, color='red', fill=False, lw=3)
ax[1].set_title('Distribution of Number of Words in Test Set', fontsize=20)
ax[1].set_xlabel('Word Number', fontsize=15)
ax[1].set_ylabel('Density', fontsize=15)
ax[1].tick_params(labelsize=12)
plt.tight_layout()
plt.show()


# Number of observations by category
hist_df = (train_set["CATEGORY_ID"]).value_counts().reset_index().rename(columns={'index': 'cat_id', "CATEGORY_ID":"obs_count"})
hist_df_50 = hist_df[:50]


plt.figure(figsize=(15, 5))
sns.barplot(data=hist_df_50, x="cat_id", y="obs_count", order=hist_df_50["cat_id"])
plt.xticks(rotation='vertical')
plt.title('Number of Observations by Category', fontsize=20)
plt.xlabel('Category', fontsize=15)
plt.ylabel('Number of Observation', fontsize=15)
plt.xticks(fontsize=12)
plt.show()


# Number of words by category (Corrected by number of observations)
word_count_df = pd.DataFrame(columns=["cat_id", "count"])
word_count_df["cat_id"] = train_set["CATEGORY_ID"]
word_count_df["word_count"] = nr_words_train
word_count_df = word_count_df.groupby(['cat_id'])['word_count'].agg("sum").sort_values(ascending=False)
word_count_df = pd.concat([pd.DataFrame(word_count_df), hist_df.set_index("cat_id")], axis=1)
word_count_df["corrected"] = word_count_df["word_count"] / word_count_df["obs_count"]
word_count_df.sort_values(by="corrected", ascending=False, inplace=True)


plt.figure(figsize=(15, 5))
sns.barplot(data=word_count_df, x=word_count_df.index, y="corrected", order=word_count_df.index, palette="viridis")
plt.xticks(rotation='vertical')
plt.title('Number of Words by Category', fontsize=20)
plt.text(112, -130, "Corrected by Number of Observations in Categories", fontsize=12, c="r")
plt.xlabel('Category', fontsize=15)
plt.ylabel('Number of Word', fontsize=15)
plt.xticks(fontsize=5)
plt.xlim(-3, 179)
plt.show()


# Word cloud for the entire dataset
plt.figure(figsize=(15, 10))
wordcloud = WordCloud(background_color="white", stopwords = stopwords.words('turkish')).generate(str(train_text_joined))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# Word cloud for 4 most frequent category
x, y = np.ogrid[:300, :300]
mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
mask = 255 * mask.astype(int)


plt.figure(figsize=(15, 10))
for row, cat_id in enumerate([1000271, 1000037, 1000833, 1001522]):
    plt.subplot(int("22"+str(row+1)))
    text = train_text_joined[train_set["CATEGORY_ID"]==cat_id]
    wordcloud = WordCloud(background_color="white", stopwords = stopwords.words('turkish'), mask=mask).generate(str(text))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("{}. Biggest Category".format(row+1), fontsize=15)
plt.tight_layout()
plt.show()


# Number of images
lengths = train_set["DESCRIPTION"].map(lambda x: len(get_img_url(x)))
lengths_df = pd.DataFrame(data=list(lengths), columns=["count"])
lengths_df = (lengths_df).value_counts().reset_index()


plt.figure(figsize=(15, 5))
sns.barplot(data=lengths_df, x="count", y=lengths_df[0], order=lengths_df["count"], palette="coolwarm")
plt.title('Frequency of Number of Images', fontsize=20)
plt.xlabel('Number of Images', fontsize=15)
plt.ylabel('Number of Observations', fontsize=15)
plt.xticks(fontsize=12)
plt.show()


# Number of images by category (Corrected by number of observations)
cat_lengths_df = pd.DataFrame(data=list(lengths), columns=["count"])
cat_lengths_df["cat_id"] = train_set["CATEGORY_ID"]
cat_lengths_df = cat_lengths_df.groupby(['cat_id'])['count'].agg("sum").sort_values(ascending=False)
cat_lengths_df = pd.concat([pd.DataFrame(cat_lengths_df), hist_df.set_index("cat_id")], axis=1)
cat_lengths_df["corrected"] = cat_lengths_df["count"] / cat_lengths_df["obs_count"]
cat_lengths_df.sort_values(by="corrected", ascending=False, inplace=True)


plt.figure(figsize=(15, 5))
sns.barplot(data=cat_lengths_df, x=cat_lengths_df.index, y="corrected", order=cat_lengths_df.index, palette="Spectral")
plt.xticks(rotation='vertical')
plt.title('Number of Images by Category', fontsize=20)
plt.text(112, -0.8, "Corrected by Number of Observations in Categories", fontsize=12, c="r")
plt.xlabel('Category', fontsize=15)
plt.ylabel('Number of Images', fontsize=15)
plt.xticks(fontsize=5)
plt.xlim(-3, 179)
plt.show()


# Train the FastText model
# Save the train set as a txt file
write_to_txt(f"train.txt", "\n".join(train_set["fasttext_text"]))


# Download the fasttext pre-trained word vectors
!wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.tr.300.vec.gz
!gzip -d cc.tr.300.vec.gz


# Train the model
fasttext_model = fasttext.train_supervised(
    input=f"train.txt",
    dim=300,
    lr=0.3,
    epoch=100,
    wordNgrams=2,
    pretrainedVectors="cc.tr.300.vec")


# Read the predictions of ML models
predictions_ml = pd.read_csv("data/tfidf_ml.csv")
predictions_ml.drop(["Unnamed: 0"], axis=1, inplace=True)
predictions_ml


# Read the InceptionV3 model predictions
predictions_img = pd.read_csv("data/inception_predictions.csv")
predictions_img.drop(["Unnamed: 0"], axis=1, inplace=True)
predictions_img.rename(columns={"DESCRIPTION": "pred"}, inplace=True)


# Ensemble model
# Add the predictions of Fasttext model
predictions_ml["fasttext"] = test_text_joined.map(lambda x: fasttext_predict(fasttext_model, x))


# Add the predictions of InceptionV3 model
predictions_ml["img"] = predictions_img["pred"].map(lambda x: image_prediction(x))


# Select the most voted category
predictions_ml["ensemble"] = predictions_ml["fasttext"]
for row in range(len(predictions_ml)):
    predictions_ml["ensemble"][row] = most_common([value for value in predictions_ml.iloc[row]])
predictions_ml.to_csv("ensemble.csv")
