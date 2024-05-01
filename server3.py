import re
import pickle

import numpy as np
from nltk.corpus import stopwords
from keras._tf_keras.keras.models import load_model
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firestore app
cred = credentials.Certificate("serviceAccountKey.json")
print(cred)
firebase_admin.initialize_app(cred)
db = firestore.client()

app = Flask(__name__)


class TextClassifier:
    def __init__(self, model_path, vectorizer_path, label_encoder_path, category_names_path):
        # Load the trained Keras model
        self.category_names = None
        self.model = load_model(model_path)

        # Load the TF-IDF vectorizer
        with open(vectorizer_path, 'rb') as file:
            self.tfidf_vectorizer = pickle.load(file)

        # Load the LabelEncoder for category labels
        with open(label_encoder_path, 'rb') as file:
            self.label_encoder = pickle.load(file)

        # Load the category names
        with open(category_names_path, 'rb') as file:
            self.category_names = pickle.load(file)

        # Define custom stopwords
        self.my_stopwords = ['jan', 'january', 'february', 'feb', 'march', 'april', 'may', 'june', 'july', 'aug',
                             'october', 'june', 'july', 'February', 'apr', 'Apr', 'february', 'jun', 'jul', 'feb',
                             'sep',
                             'august', 'sept', 'september', 'oct', 'october', 'nov', 'november', 'dec', 'december',
                             'mar',
                             'november october', 'wasnt']

    def clean_and_tokenize(self, text):
        # Remove URLs
        cleaned_text = re.sub(r'http\S+', '', text)
        # Remove special characters, numbers, and convert to lowercase
        cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
        cleaned_text = re.sub(r'\d+', '', cleaned_text)
        cleaned_text = cleaned_text.lower()
        # Tokenize and remove stopwords (including custom stopwords)
        stop = stopwords.words('english')
        combined_stopwords = stop + self.my_stopwords
        tokens = [word for word in cleaned_text.split() if word not in combined_stopwords]
        return ' '.join(tokens)

    def predict_category(self, text):
        # Preprocess the input text
        cleaned_text = self.clean_and_tokenize(text)

        # Transform preprocessed text into TF-IDF features
        features = self.tfidf_vectorizer.transform([cleaned_text]).toarray()

        # Reshape features for model input (assuming 1D convolution)
        model_input = features.reshape((1, features.shape[1]))

        # Make predictions using the trained model
        predictions = self.model.predict(model_input)
        # Get predicted category label (assuming 'model' outputs softmax probabilities)
        predicted_category_index = np.argmax(predictions)
        predicted_category = self.category_names[predicted_category_index]

        return predicted_category


def retrieve_data(collection_name):
    """
    Retrieve data from a Firestore collection.

    Args:
        collection_name (str): The name of the collection to retrieve data from.

    Returns:
        list: A list of dictionaries containing the retrieved documents.
    """
    try:
        # Get all documents from the specified collection
        docs = db.collection(collection_name).get()
        data = []
        # Iterate through the documents and append them to the data list
        for doc in docs:
            data.append(doc.to_dict())
        return data
    except Exception as e:
        print("An error occurred:", e)
        return []


def parse_data(document: list):
    data = []
    for doc in document:
        text = doc['test']
        data_point = {'text': text}
        data.append(data_point)

    return data


@app.route('/predict', methods=['GET'])
def predict():
    try:

        collection_name = "CommentTest"
        document = retrieve_data(collection_name)
        data = parse_data(document)

        # Initialize an empty list to store the prediction results
        results = []

        # Initialize an empty dictionary to store category counts
        category_counts = {
            "Bug": 0,
            "comments": 0,
            "complaints": 0,
            "meaningless": 0,
            "requests": 0
        }

        # Process each text object in the input JSON data
        for item in data:
            if 'text' in item:
                text = item['text']
                # Perform prediction using TextClassifier
                prediction_result = classifier.predict_category(text)
                print(prediction_result)

                # Append predicted category to result along with input text
                result = {
                    "text": text,
                    "category": prediction_result
                }
                db.collection("prediction").add(result)
                results.append(result)
                # Increment the count for the predicted category
                category_counts[prediction_result] += 1

        db.collection('results').add(category_counts)
        # Return the prediction results as JSON response

        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# @app.route('/get_results', methods=['GET'])
# def get_results():
#     # Assuming `results` is the variable containing your data
#     try:
#         # Parse the JSON data from the request
#         data = request.get_json()
#
#         # Initialize an empty dictionary to store category counts
#         category_counts = {
#             "Bug": 0,
#             "comments": 0,
#             "complaints": 0,
#             "meaningless": 0,
#             "requests": 0
#         }
#
#         # Process each text object in the input JSON data
#         for item in data:
#             if 'text' in item:
#                 text = item['text']
#                 # Perform prediction using TextClassifier
#                 prediction_result = classifier.predict_category(text)
#
#                 # Increment the count for the predicted category
#                 category_counts[prediction_result] += 1
#
#         db.collection('results').add(category_counts)
#         # Return the category counts as JSON response
#         return jsonify(category_counts), 200
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# Example usage:
if __name__ == "__main__":
    # Initialize TextClassifier with paths to model and vectorizer
    model_path = 'text_classifier.keras'
    vectorizer_path = 'tfidf_vectorizer.pkl'
    label_encoder_path = 'label_encoder.pkl'
    category_names_path = 'category_names.pkl'
    classifier = TextClassifier(model_path, vectorizer_path, label_encoder_path, category_names_path)

    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5001)
