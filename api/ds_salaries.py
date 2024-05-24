import sys
sys.path.append('/home/sumi/vscode/flask-portfolio-1')
from flask import Blueprint, Flask, request, jsonify
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
import pandas as pd
<<<<<<< HEAD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
=======
from model.salaries_model import SalaryModel
>>>>>>> 3e91df6ee50f35d1bb61c9ad23d59840990f5bb7

app = Flask(__name__)
salaries_api = Blueprint('salaries_api', __name__, url_prefix='/api/salaries')
api = Api(salaries_api)
CORS(salaries_api)

<<<<<<< HEAD
class SalaryModel:

    _instance = None

    def __init__(self):
        self.model = None
        self.features = ['work_year', 'experience_level', 'employment_type', 'job_title', 'salary_in_usd', 'remote_ratio']
        self.target = 'salary'
        self.salary_data = pd.read_csv("/home/sumi/vscode/flask-portfolio-1/ds_salaries.csv")
        self.encoder = OneHotEncoder(handle_unknown='ignore')

        self._clean()
        self._train()

    def _clean(self):
        self.salary_data.dropna(inplace=True)

    def _train(self):
        X = self.salary_data[self.features].copy()  # Make a copy of the DataFrame
        y = self.salary_data[self.target]
        self.encoder.fit(X[['experience_level', 'employment_type', 'job_title']])

        X_encoded = self.encoder.transform(X[['experience_level', 'employment_type', 'job_title']]).toarray()
        X = X.drop(['experience_level', 'employment_type', 'job_title'], axis=1)
        X = pd.concat([X, pd.DataFrame(X_encoded)], axis=1)

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def predict_salary(self, input_data):
        try:
            input_features = pd.DataFrame([input_data], columns=self.features)
            input_features_encoded = self.encoder.transform(input_features[['experience_level', 'employment_type', 'job_title']]).toarray()
            input_features = input_features.drop(['experience_level', 'employment_type', 'job_title'], axis=1)
            input_features = pd.concat([input_features, pd.DataFrame(input_features_encoded)], axis=1)
            input_features.columns = input_features.columns.astype(str)

            predicted_salary = self.model.predict(input_features)[0]
            return {'predicted_salary': float(predicted_salary)}, 200
        except Exception as e:
            return {'error': str(e)}, 400

class SalaryAPI:
    class _Predict(Resource):
        def post(self):
            try:
                parser = reqparse.RequestParser()
                parser.add_argument('work_year', type=str, required=True)
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
=======
class Predict(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('work_year', type=str, required=True)
            parser.add_argument('experience_level', type=str, required=True)
            parser.add_argument('employment_type', type=str, required=True)
            parser.add_argument('job_title', type=str, required=True)
            parser.add_argument('salary_currency', type=str, required=True)
            parser.add_argument('salary_in_usd', type=str, required=True)
            parser.add_argument('employee_residence', type=str, required=True)
            parser.add_argument('remote_ratio', type=str, required=True)
            parser.add_argument('company_location', type=str, required=True)
            parser.add_argument('company_size', type=str, required=True)
            args = parser.parse_args()
>>>>>>> 3e91df6ee50f35d1bb61c9ad23d59840990f5bb7

            salary_model = SalaryModel.get_instance()
            response, status_code = salary_model.predict(args)
            return response, status_code
        except Exception as e:
            return {'error': str(e)}, 400

api.add_resource(Predict, '/predict')

<<<<<<< HEAD
                print("Mapped experience level:", mapped_experience_level)

                if mapped_experience_level is None:
                    raise ValueError("Invalid experience level")

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
                job_title = args['job_title'].strip().lower()
                job_title_id = job_title_mapping.get(job_title)

                print("Mapped job title:", job_title_id)

                if job_title_id is None:
                    raise ValueError("Invalid job title")

                salary_data = {
                    'work_year': args['work_year'],
                    'experience_level': mapped_experience_level,
                    'employment_type': args['employment_type'],
                    'job_title': job_title_id,
                    'salary_currency': args['salary_currency'],
                    'salary_in_usd': args['salary_in_usd'],
                    'employee_residence': args['employee_residence'],
                    'remote_ratio': args['remote_ratio'],
                    'company_location': args['company_location'],
                    'company_size': args['company_size']
                }

                salary_model_instance = SalaryModel.get_instance()
                response, status_code = salary_model_instance.predict_salary(salary_data)

                return jsonify(response), status_code
            except Exception as e:
                return {'error': str(e)}, 400

api.add_resource(SalaryAPI._Predict, '/predict')

if __name__ == '__main__':
=======
if __name__ == "__main__":
>>>>>>> 3e91df6ee50f35d1bb61c9ad23d59840990f5bb7
    app.register_blueprint(salaries_api)
    app.run(debug=True)