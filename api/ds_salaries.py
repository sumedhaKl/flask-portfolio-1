from flask import Flask
import pandas as pd
from flask import Blueprint
from flask_restful import Api, Resource, reqparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
salaries_api = Blueprint('salaries_api', __name__, url_prefix='/api/salaries')
api = Api(salaries_api)

class Predict(Resource):
    def post(self):
        try:
            # Parse incoming request data
            parser = reqparse.RequestParser()
            parser.add_argument('job_title', type=str, required=True)
            parser.add_argument('experience_level', type=str, required=True)
            args = parser.parse_args()
            
            # Read CSV data
            salary_data = pd.read_csv("ds_salaries.csv")
            
            # Preprocessing
            salary_data['job_title'] = args['job_title']
            salary_data['experience_level'] = args['experience_level']
            salary_data['job_title'] = salary_data['job_title'].apply(lambda x: 1 if x else 0)
            salary_data['experience_level'] = salary_data['experience_level'].apply(lambda x: 1 if x else 0)
            enc = OneHotEncoder(handle_unknown='ignore')
            onehot = enc.transform(salary_data[['job_title', 'experience_level']]).toarray()
            cols = ['job_title_' + val for val in enc.categories_[0]]
            cols += ['experience_level_' + val for val in enc.categories_[1]]
            salary_data[cols] = pd.DataFrame(onehot)
            salary_data.drop(['job_title', 'experience_level'], axis=1, inplace=True)
            
            # Predict the salary probability for the new data
            logreg = LogisticRegression()
            salary_prob = logreg.predict_proba(salary_data)[:, 1]

            return {'salary_probability': float(salary_prob)}
        except Exception as e:
            return {'error': str(e)}, 400

api.add_resource(Predict, '/predict')