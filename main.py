import glob
import json
import os
import re
from datetime import datetime
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from werkzeug.utils import secure_filename

import dotenv
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, Response
import joblib
import nltk

nltk.download('stopwords')

app = Flask(__name__)

model = None

# dotenv.load_dotenv('.env')

_threshold = float(os.getenv("THRESHOLD"))

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['csv']
app.config['UPLOAD_PATH'] = '/app/data/models'
app.config['DATASET_DIR_PATH'] = '/app/data/datasets'

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwords from text
    return text


def load_model():
    model_files = glob.glob('/app/data/models/model-*.joblib')
    if not model_files:
        raise FileNotFoundError("No model files found.")

    sorted_model_files = sorted(model_files, key=os.path.getmtime)

    latest_model_file = sorted_model_files[-1]

    global model
    model = joblib.load(latest_model_file)


def get_latest_dataset():
    dataset_files = glob.glob('/app/data/datasets/dataset-*.csv')
    if not dataset_files:
        raise FileNotFoundError("No model files found.")

    sorted_dataset_files = sorted(dataset_files, key=os.path.getmtime)

    latest_dataset_file = sorted_dataset_files[-1]

    return latest_dataset_file


@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():
    if 'dataset' not in request.files:
        return 'No file part in the request'

    file = request.files['dataset']

    if file.filename == '':
        return 'No file selected'

    if file.filename.rsplit('.', 1)[1].lower() not in app.config['UPLOAD_EXTENSIONS']:
        return 'Invalid file extension'

    now = datetime.now()
    formatted_date = now.strftime("%d-%m-%y")

    filename = secure_filename(file.filename)
    file.save(f"{app.config['DATASET_DIR_PATH']}/dataset-{formatted_date}.csv")

    return 'File saved successfully'


def predict_with_fallback(model_repr, input_data_param, threshold_param, fallback_value):
    proba = model_repr.predict_proba(input_data_param)

    max_proba = np.max(proba, axis=1)

    if np.any(max_proba > threshold_param):
        predictions = model_repr.predict(input_data_param)
        return predictions[0]

    else:
        return fallback_value


@app.route("/ping", methods=["GET"])
def test():
    return jsonify({"message": "Pong!"})


@app.route("/train_model", methods=["GET"])
def new_model():
    global model

    df = pd.read_csv(get_latest_dataset(), sep=';')
    df = df[pd.notnull(df['InputData'])]

    df['InputData'] = df['InputData'].apply(clean_text)

    X = df.InputData
    y = df.Function

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

    recent_model = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB()),
                             ])
    recent_model.fit(X_train, y_train)

    now = datetime.now()
    formatted_date = now.strftime("%d-%m-%y")

    joblib.dump(recent_model, filename=f'./models/model-{formatted_date}.joblib')

    model = recent_model

    json_data = {
        "success": True,
        "message": "Model trained and saved successfully"
    }

    return Response(
        json.dumps(json_data),
        status=200,
        mimetype="application/json"
    )

def create_model(dataset):
    df = pd.read_csv(dataset, sep=';')
    df = df[pd.notnull(df['InputData'])]

    df['InputData'] = df['InputData'].apply(clean_text)

    X = df.InputData
    y = df.Function

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

    recent_model = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB()),
                             ])
    recent_model.fit(X_train, y_train)

    now = datetime.now()
    formatted_date = now.strftime("%d-%m-%y")

    joblib.dump(recent_model, filename=f'./models/model-{formatted_date}.joblib')


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    user_input = data['user_input']

    if not user_input:
        json_data = {
            "success": False,
            "message": ""
        }

        return Response(
            json.dumps(json_data),
            status=500,
            mimetype="application/json"
        )

    prediction = predict_with_fallback(model, [user_input], _threshold, "General.Output.UnknownCommand")

    json_data = {
        "success": True,
        "message": prediction
    }

    return Response(
        json.dumps(json_data),
        status=200,
        mimetype="application/json"
    )


if __name__ == '__main__':
    try:
        os.makedirs(name="/app/data/models")
    except:
        pass
    try:
        os.makedirs(name="/app/data/datasets")
    except:
        pass
    create_model('/app/data/init.csv')

    load_model()
    app.run(debug=False, host='0.0.0.0', port=5000)
