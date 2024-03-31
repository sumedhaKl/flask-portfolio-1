from flask import Flask
import pandas as pd
from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from model.titanic_model import TitanicModel

app = Flask(__name__)
titanic_api = Blueprint('titanic_api', __name__, url_prefix='/api/titanic')
api = Api(titanic_api)

class TitanicAPI:
    class _Predict(Resource):
        
        def post(self):
            
            passenger = request.get_json()

            # Get the singleton instance of the TitanicModel
            titanicModel = TitanicAPI.get_instance()
            # Predict the survival probability of the passenger
            response = titanicModel.predict(passenger)

            # Return the response as JSON
            return jsonify(response)

#Checks Titanic model
model = TitanicModel

class Predict(Resource):
    def post(self):
        data = request.get_json()
        passenger_data = pd.DataFrame(data, index=[0])
            
            #Returns from Titanic model api to frontend
        return model.Predict(passenger_data)

api.add_resource(Predict, '/predict')