
"""
from flask import Flask, Blueprint
from flask_restful import Api, Resource, reqparse
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
salaries_api = Blueprint('salaries_api', __name__, url_prefix='/api/salaries')
api = Api(salaries_api)

class SalaryModel:
    _instance = None

    def __init__(self):
        self.model = None
        self.encoder = OneHotEncoder(handle_unknown='ignore')

    def _clean(self):
        # Load and preprocess data
        self.salary_data = pd.read_csv("ds_salaries.csv")
        self.salary_data['work_year'] = self.salary_data['work_year'].apply(lambda x: 1 if x else 0)
        self.salary_data['experience_level'] = self.salary_data['experience_level'].map({'entry': 'EN', 'mid': 'MI', 'senior': 'SE', 'expert': 'EX'})
        self.salary_data['employment_type'] = self.salary_data['employment_type'].map({'ft': 1})  # Assuming 'ft' for full-time as the only option
        self.salary_data['job_title'] = self.salary_data['job_title'].map({
            'Data Scientist': 1, 'Data Analyst': 2, 'Data Engineer': 3, 'Machine Learning Scientist': 4,
            'Big Data Engineer': 5, 'Product Data Analyst': 6, 'Machine Learning Engineer': 7,
            'Lead Data Scientist': 8, 'Business Data Analyst': 9, 'Lead Data Engineer': 10,
            'Lead Data Analyst': 11, 'Data Scientist Consultant': 12, 'BI Data Analyst': 13,
            'Director of Data Science': 14, 'Research Scientist': 15, 'Machine Learning Manager': 16,
            'Data Engineering Manager': 17, 'Machine Learning Infrastructure Engineer': 18, 'ML Engineer': 19,
            'AI Scientist': 20, 'Computer Vision Engineer': 21, 'Principal Data Scientist': 22,
            'Head of Data': 23, '3D Computer Vision Researcher': 24, 'Applied Data Scientist': 25,
            'Marketing Data Analyst': 26, 'Cloud Data Engineer': 27, 'Financial Data Analyst': 28,
            'Computer Vision Software Engineer': 29, 'Data Science Manager': 30, 'Data Analytics Engineer': 31,
            'Applied Machine Learning Scientist': 32, 'Data Specialist': 33, 'Data Science Engineer': 34,
            'Big Data Architect': 35, 'Head of Data Science': 36, 'Analytics Engineer': 37,
            'Data Architect': 38, 'Head of Machine Learning': 39, 'ETL Developer': 40,
            'Lead Machine Learning Engineer': 41, 'Machine Learning Developer': 42, 'Principal Data Analyst': 43,
            'NLP Engineer': 44, 'Data Analytics Lead': 45
        })
        self.salary_data['salary_currency'] = self.salary_data['salary_currency'].map({'USD': 1})  # Assuming USD as the only currency
        self.salary_data['salary_in_usd'] = self.salary_data['salary_in_usd']  # Assuming 'salary' column is in USD
        self.encoder.fit(self.salary_data[['work_year', 'experience_level', 'employment_type', 'job_title', 'salary_currency']])

    def _train(self):
        # Train the model
        X = self.salary_data[['work_year', 'experience_level', 'employment_type', 'job_title', 'salary_currency']]
        y = self.salary_data['salary_in_usd']
        self.model = LogisticRegression()
        self.model.fit(X, y)

    def predict(self, data):
        # Predict salary probability
        work_year = data['work_year']
        experience_level = data['experience_level']
        employment_type = data['employment_type']
        job_title = data['job_title']
        salary_currency = data['salary_currency']

        input_data = pd.DataFrame([[work_year, experience_level, employment_type, job_title, salary_currency]], columns=['work_year', 'experience_level', 'employment_type', 'job_title', 'salary_currency'])
        input_data[['work_year', 'experience_level', 'employment_type', 'job_title', 'salary_currency']] = self.encoder.transform(input_data[['work_year', 'experience_level', 'employment_type', 'job_title', 'salary_currency']])
        salary_probability = self.model.predict_proba(input_data)[:, 1]
        return float(salary_probability)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._clean()
            cls._instance._train()
        return cls._instance

class Predict(Resource):
    def post(self):
        try:
            # Parse incoming request data
            parser = reqparse.RequestParser()
            parser.add_argument('work_year', type=int, required=True)
            parser.add_argument('experience_level', type=str, required=True)
            parser.add_argument('employment_type', type=float, required=True)
            parser.add_argument('job_title', type=int, required=True)
            parser.add_argument('salary_currency', type=str, required=True)
            args = parser.parse_args()

            # Get singleton instance of SalaryModel
            salary_model = SalaryModel.get_instance()

            # Predict salary
            salary_prob = salary_model.predict(args)

            return {'salary_probability': salary_prob}
        except Exception as e:
            return {'error': str(e)}, 400

api.add_resource(Predict, '/predict')

if __name__ == "__main__":
    app.register_blueprint(salaries_api)
    app.run(debug=True)
    """