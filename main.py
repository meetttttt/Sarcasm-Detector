# Importing the necessary library

import re
import string
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.metrics import ConfusionMatrixDisplay

# Loading the data

data = pd.read_json(
    "/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json"
    , lines=True)

# Checking the shape of the data

print(data.shape)

# Displaying the first ten rows of the dataset

print(data.head(10))

# Checking for null values

print(data.isnull().sum())

# Checking for duplicate values

print(data.duplicated().sum())

# Too be continued ...
