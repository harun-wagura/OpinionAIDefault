#importing libraries
import pandas as pd
import numpy as np
import os
import pickle

#For visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For Transforming our target vatiable
from keras._tf_keras.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

#For preprocessing text data
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

path1 = "./Data/bugs.txt"
print(path1)
path2 = './Data/comments.txt'
print(path2)
path3 = './Data/complaints.txt'
print(path3)
path4 = './Data/meaningless.txt'
print(path4)
path5 = './Data/requests.txt'
print(path5)


def text_data(path):
    text_body_appended = []
    with open(path, "r", encoding='windows-1256') as f:
        for line in f:
            text_body_appended.append(line.strip())
    return text_body_appended


bugs = text_data(path1)
comments = text_data(path2)
complaints = text_data(path3)
meaningless = text_data(path4)
requests = text_data(path5)

print(len(bugs))
print(len(comments))
print(len(complaints))
print(len(meaningless))
print(len(requests))


def data_frame(txt, category):
    column_names = ('text', 'Category')
    df = pd.DataFrame(columns=column_names)
    df['text'] = txt
    df['Category'] = category
    return df


data_list = []
categories = ["Bug", "comments", "complaints", "meaningless", "requests"]
text_data_sources = [bugs, comments, complaints, meaningless, requests]

for category, text_data in zip(categories, text_data_sources):
    data_list.append(data_frame(text_data, category))
data = pd.concat(data_list)

data['Category'].unique()
data.head()
data.groupby(['Category']).size()

#remove link starts with https
data['text'] = data['text'].map(lambda x: re.sub('http.*', '', str(x)))
#removing data and time (numeric values)
data['text'] = data['text'].map(lambda x: re.sub('[0-9]', '', str(x)))
#removing \n
data['text'] = data['text'].map(lambda x: re.sub('[\\n]', '', str(x)))
#removing some special characters
data['text'] = data['text'].map(lambda x: re.sub('[#|*|$|:|\\|&]', '', str(x)))

my_stopwords = ['jan', 'january', 'february' 'feb', 'march', 'april', 'may', 'june', 'july', 'aug',
                'october', 'October', 'june', 'july', 'February', 'apr', 'Apr', 'february', 'jun', 'jul', 'feb', 'sep',
                'august', 'sept', 'september', 'oct', 'october', 'nov', 'november', 'dec', 'december', 'mar',
                'november october', 'wasnt']


def clean_and_tokenize(text):
    # Perform text cleaning steps discussed earlier (URL removal, etc.)
    cleaned_text = text  # Assuming no separate cleaning step needed
    # Combine stopwords and remove them from the cleaned text
    stop = stopwords.words('english')
    combined_stopwords = stop + my_stopwords
    tokens = [word for word in cleaned_text.split() if word not in combined_stopwords]

    return ' '.join(tokens)  # Join the tokens into a string

# Clean and tokenize text data
data['cleaned_text'] = data['text'].apply(clean_and_tokenize)

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

# Fit TF-IDF vectorizer and transform text data into TF-IDF features
x_features = tfidf_vectorizer.fit_transform(data['cleaned_text']).toarray()

# Assuming 'tfidf_vectorizer' is your fitted TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(tfidf_vectorizer, file)

x_features = pd.DataFrame(x_features)

#preparing target variable
target = data['Category']
label = LabelEncoder()
target = label.fit_transform(target)
target = to_categorical(target)

target = pd.DataFrame(data=target, columns=['Bug', 'comments', 'complaints', 'meaningless', 'requests'])
target.head()


# IMPLEMENTED Stratified KFold instead
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression as lg

# Define the Logistic Regression model with One-vs-Rest multi-class strategy
logistic = lg(penalty='l2', solver='newton-cg', C=5, multi_class='ovr', max_iter=5000)

# Perform Stratified KFold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True)  # 5 folds with shuffling

# List to store accuracy scores for each category
category_wise_accuracy = []

# Loop through each target category (assuming target is a DataFrame)
for i in range(target.shape[1]):
    # Extract target labels for the current category
    y_target = target.iloc[:, i]

    # Perform cross-validation for the current category
    for train_index, test_index in cv.split(x_features, y_target):
        X_train, X_test = x_features.iloc[train_index], x_features.iloc[test_index]
        y_train, y_test = y_target.iloc[train_index], y_target.iloc[test_index]

        # Train the model on the training data
        ovr_classifier = OneVsRestClassifier(estimator=logistic)
        ovr_classifier.fit(X_train, y_train)

        # Evaluate the model on the testing data
        accuracy = ovr_classifier.score(X_test, y_test)
        category_wise_accuracy.append(accuracy)

# Print the average accuracy for each category
for i in range(target.shape[1]):
    print(f"Average Accuracy (Category {target.columns[i]}):", sum(category_wise_accuracy[i * 5:i * 5 + 5]) / 5)

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout
from keras import regularizers

#preparing target variable
target = data['Category']
label = LabelEncoder()
target = label.fit_transform(target)

# Get the unique category names corresponding to the encoded labels
category_names = label.classes_

# Save the LabelEncoder and category names to a file using pickle
label_encoder_path = 'label_encoder.pkl'
category_names_path = 'category_names.pkl'

# Save LabelEncoder
with open(label_encoder_path, 'wb') as file:
    pickle.dump(label, file)

# Save category names
with open(category_names_path, 'wb') as file:
    pickle.dump(category_names, file)


target = to_categorical(target)

target = pd.DataFrame(data=target, columns=['Bug', 'comments', 'complaints', 'meaningless', 'requests'])
target.head()

clf = Sequential()
clf.add(Dense(units=2048, activation="relu", kernel_initializer="uniform", kernel_regularizer=regularizers.l2(0.001)))
clf.add(Dropout(0.2))
clf.add(Dense(units=2048, activation="relu", kernel_initializer="uniform", kernel_regularizer=regularizers.l2(0.001)))
clf.add(Dropout(0.2))
clf.add(Dense(units=2048, activation="relu", kernel_initializer="uniform", kernel_regularizer=regularizers.l2(0.001)))
clf.add(Dropout(0.2))
clf.add(Dense(units=5, activation="softmax", kernel_initializer="uniform"))
clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

hist = clf.fit(x_features, target, batch_size=32, epochs=24)

clf.save('text_classifier.keras')


