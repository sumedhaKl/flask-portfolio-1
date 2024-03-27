from flask import Flask
import pandas as pd
from flask import Blueprint, request
from flask_restful import Api, Resource
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from model.titanic_model import predict as predict_model

app = Flask(__name__)
titanic_api = Blueprint('titanic_api', __name__, url_prefix='/api/titanic')
api = Api(titanic_api)


#Checks Titanic model
model = predict_model



class Predict(Resource):
    def post(self):
        try:
            data = request.get_json()
            passenger_data = pd.DataFrame(data, index=[0])
            
            #Returns from Titanic model api to frontend
            return model.Predict(passenger_data)
            
            return {'death_percentage': float(death_prob * 100), 'survivability_percentage': float(survival_prob * 100)}, 200
        except Exception as e:
            return {'error': str(e)}, 400

api.add_resource(Predict, '/predict')