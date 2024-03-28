from flask import Flask
import pandas as pd
from flask import Blueprint
from flask_restful import Api, Resource
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
salaries_api = Blueprint('slaries_api', __name__, url_prefix='/api/salaries')
api = Api(salaries_api)

class Predict(Resource):
    def post(self):
        try:
            salary_data = pd.read_csv("ds_salaries.csv")
            
            # Preprocesssing
            salary_data['sex'] = salary_data['sex'].apply(lambda x: 1 if x == 'male' else 0)
            salary_data['alone'] = salary_data['alone'].apply(lambda x: 1 if x else 0)
            enc = OneHotEncoder(handle_unknown='ignore')
            onehot = enc.transform(salary_data[['embarked']]).toarray()
            cols = ['embarked_' + val for val in enc.categories_[0]]
            salary_data[cols] = pd.DataFrame(onehot)
            salary_data.drop(['embarked'], axis=1, inplace=True)
            
            # Predict the survival probability for the new passenger
            logreg = LogisticRegression()
            survival_prob = logreg.predict_proba(salary_data)[:, 1]
            death_prob = 1 - survival_prob

            return {'death_percentage': float(death_prob * 100), 'survivability_percentage': float(survival_prob * 100)}, 200
        except Exception as e:
            return {'error': str(e)}, 400

api.add_resource(Predict, '/predict')