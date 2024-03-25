from flask import Flask
import pandas as pd
from flask import Blueprint, request
from flask_restful import Api, Resource
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
<<<<<<< HEAD
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
=======
from model.titanic_model import predict_model
>>>>>>> 4d7002c4b6a690d3597749d05d69c8c0fce80d3e

app = Flask(__name__)
titanic_api = Blueprint('titanic_api', __name__, url_prefix='/api/titanic')
api = Api(titanic_api)

<<<<<<< HEAD
titanic_data = sns.load_dataset('titanic')
=======

#Checks Titanic model
model = predict_model()


>>>>>>> 4d7002c4b6a690d3597749d05d69c8c0fce80d3e

class Predict(Resource):
    def post(self):
        try:
            data = request.get_json()
            passenger_data = pd.DataFrame(data, index=[0])
            
<<<<<<< HEAD
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
=======
            #Returns from Titanic model api to frontend
            return model.Predict()
            
            return {'death_percentage': float(death_prob * 100), 'survivability_percentage': float(survival_prob * 100)}, 200
>>>>>>> 4d7002c4b6a690d3597749d05d69c8c0fce80d3e
        except Exception as e:
            return {'error': str(e)}, 400

api.add_resource(Predict, '/predict')