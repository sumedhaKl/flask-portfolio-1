from flask import Blueprint, Flask, request, jsonify
from flask_restful import Api, Resource, reqparse
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from flask_cors import CORS

app = Flask(__name__)
salaries_api = Blueprint('salaries_api', __name__, url_prefix='/api/salaries')
api = Api(salaries_api)
CORS(salaries_api)

class SalaryModel:
    _instance = None

    def __init__(self):
        self.model = None
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self._clean()
        self._train()

    def _clean(self):
        csv_file_path = '/home/sumi/vscode/flask-portfolio-1/ds_salaries.csv'
        self.salary_data = pd.read_csv(csv_file_path)
        self.salary_data.dropna(inplace=True)
        self.salary_data['work_year'] = pd.to_numeric(self.salary_data['work_year'])
        self.salary_data['salary_in_usd'] = pd.to_numeric(self.salary_data['salary_in_usd'])
        self.salary_data['remote_ratio'] = pd.to_numeric(self.salary_data['remote_ratio'])
        self.features = ['work_year', 'experience_level', 'employment_type', 'job_title', 'salary_in_usd', 'remote_ratio']
        self.encoder.fit(self.salary_data[['experience_level', 'employment_type', 'job_title']])

    def _train(self):
        X = self.salary_data[self.features].copy()
        X_encoded = self.encoder.transform(X[['experience_level', 'employment_type', 'job_title']]).toarray()
        X = X.drop(['experience_level', 'employment_type', 'job_title'], axis=1)
        X = pd.concat([X, pd.DataFrame(X_encoded)], axis=1)
        y = self.salary_data['salary_in_usd']
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)

    def predict(self, data):
        try:
            experience_level_mapping = {
                'entry': 'EN',
                'mid': 'MI',
                'senior': 'SE',
                'expert': 'EX'
            }
            job_title_mapping = {
                'data scientist': 1,
                'data analyst': 2,
                'data engineer': 3,
                'machine learning scientist': 4,
                'big data engineer': 5,
                'product data analyst': 6,
                'machine learning engineer': 7,
                'lead data scientist': 8,
                'business data analyst': 9,
                'lead data engineer': 10,
                'lead data analyst': 11,
                'data scientist consultant': 12,
                'bi data analyst': 13,
                'director of data science': 14,
                'research scientist': 15,
                'machine learning manager': 16,
                'data engineering manager': 17,
                'machine learning infrastructure engineer': 18,
                'ml engineer': 19,
                'ai scientist': 20,
                'computer vision engineer': 21,
                'principal data scientist': 22,
                'head of data': 23,
                '3d computer vision researcher': 24,
                'applied data scientist': 25,
                'marketing data analyst': 26,
                'cloud data engineer': 27,
                'financial data analyst': 28,
                'computer vision software engineer': 29,
                'data science manager': 30,
                'data analytics engineer': 31,
                'applied machine learning scientist': 32,
                'data specialist': 33,
                'data science engineer': 34,
                'big data architect': 35,
                'head of data science': 36,
                'analytics engineer': 37,
                'data architect': 38,
                'head of machine learning': 39,
                'etl developer': 40,
                'lead machine learning engineer': 41,
                'machine learning developer': 42,
                'principal data analyst': 43,
                'nlp engineer': 44,
                'data analytics lead': 45
            }

            experience_level = experience_level_mapping.get(data['experience_level'].strip().lower())
            if experience_level is None:
                raise ValueError("Invalid experience level")

            job_title = job_title_mapping.get(data['job_title'].strip().lower())
            if job_title is None:
                raise ValueError("Invalid job title")

            input_data = pd.DataFrame([[
                data['work_year'],
                experience_level,
                data['employment_type'],
                job_title,
                data['salary_in_usd'],
                data['remote_ratio']
            ]], columns=self.features)
            
            input_data_encoded = self.encoder.transform(input_data[['experience_level', 'employment_type', 'job_title']]).toarray()
            input_data = input_data.drop(['experience_level', 'employment_type', 'job_title'], axis=1)
            input_data = pd.concat([input_data, pd.DataFrame(input_data_encoded)], axis=1)
            input_data.columns = input_data.columns.astype(str)

            salary_probability = self.model.predict(input_data)[0]
            return {'predicted_salary': float(salary_probability)}, 200
        except (ValueError, KeyError) as e:
            return {'error': str(e)}, 400

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

<<<<<<< HEAD
class Predict(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('work_year', type=int, required=True)
            parser.add_argument('experience_level', type=str, required=True)
            parser.add_argument('employment_type', type=str, required=True)
            parser.add_argument('job_title', type=str, required=True)
            parser.add_argument('salary_currency', type=str, required=True)
            parser.add_argument('salary_in_usd', type=float, required=True)
            parser.add_argument('employee_residence', type=str, required=True)
            parser.add_argument('remote_ratio', type=float, required=True)
            parser.add_argument('company_location', type=str, required=True)
            parser.add_argument('company_size', type=str, required=True)
            args = parser.parse_args()

            salary_model = SalaryModel.get_instance()

            response, status_code = salary_model.predict(args)
            return response, status_code
        except Exception as e:
            return {'error': str(e)}, 400

api.add_resource(Predict, '/predict')

if __name__ == "__main__":
    app.register_blueprint(salaries_api)
    app.run(debug=True)
=======
                return jsonify(response), status_code
            except Exception as e:
                return {'error': str(e)}, 400
>>>>>>> 535a8a7c1d3df908b6f8ce29cbeb7950c7af8238
