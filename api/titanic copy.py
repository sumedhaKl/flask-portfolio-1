from flask import Flask, request, jsonify
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train_model():
    # Load the titanic dataset
    titanic_data = sns.load_dataset('titanic')

    # Preprocess the data
    td = preprocess_data(titanic_data)

    # Build distinct data frames on survived column
    X, y = split_data(td)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a decision tree classifier
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train) 

    # Train a logistic regression model
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    return 'Models trained successfully'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print(titanic_data.median())
    print(titanic_data.query("survived == 0").mean())
    print(td.query("survived == 1").mean())
    print("maximums for survivors")
    print(td.query("survived == 1").max())
    print()
    print("minimums for survivors")
    print(td.query("survived == 1").min())
    # Load the trained models
    # Here you would load the trained models from disk or memory

    # Make predictions
    # Here you would use the loaded models to make predictions on new data

    return jsonify(predict)

def preprocess_data(data):
    # Preprocess the data
    td = data.copy()
    td.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'], axis=1, inplace=True)
    td.dropna(inplace=True)
    td['sex'] = td['sex'].apply(lambda x: 1 if x == 'male' else 0)
    td['alone'] = td['alone'].apply(lambda x: 1 if x is True else 0)

    # Encode categorical variables
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(td[['embarked']])
    onehot = enc.transform(td[['embarked']]).toarray()
    cols = ['embarked_' + val for val in enc.categories_[0]]
    td[cols] = pd.DataFrame(onehot)
    td.drop(['embarked'], axis=1, inplace=True)
    td.dropna(inplace=True)

    return td

def split_data(data):
    X = data.drop('survived', axis=1)
    y = data['survived']
    return X, y

if __name__ == '__main__':
    app.run(debug=True)
