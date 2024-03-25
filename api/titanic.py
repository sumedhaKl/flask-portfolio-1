from flask import Flask
import pandas as pd
from flask import Blueprint, request
from flask_restful import Api, Resource
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score

app = Flask(__name__)
titanic_api = Blueprint('titanic_api', __name__, url_prefix='/api/titanic')
api = Api(titanic_api)

titanic_data = sns.load_dataset('titanic')

class Predict(Resource):
    def post(self):
        try:
            data = request.get_json()
            passenger_data = pd.DataFrame(data, index=[0])
            
            # Preprocesssing
            titanic_data.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'], axis=1, inplace=True)
            titanic_data.dropna(inplace=True)
            titanic_data['sex'] = titanic_data['sex'].apply(lambda x: 1 if x == 'male' else 0)
            titanic_data['alone'] = titanic_data['alone'].apply(lambda x: 1 if x == True else 0)
            
            X = titanic_data.drop('survived', axis=1)
            y = titanic_data['survived']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            dt = DecisionTreeClassifier()
            dt.fit(X_train, y_train)
            v_ = dt.predict(passenger_data)
            print(v_)

            # Predict the survival probability for the new passenger

            return {v_}, 200
        except Exception as e:
            return {'error': str(e)}, 400

api.add_resource(Predict, '/predict')