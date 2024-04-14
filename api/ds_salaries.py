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
        self.salary_data['work_year'] = pd.to_numeric(self.salary_data['work_year'], errors='coerce')
        self.salary_data['salary'] = pd.to_numeric(self.salary_data['salary'])
        self.salary_data['salary_in_usd'] = pd.to_numeric(self.salary_data['salary_in_usd'])
        self.salary_data['remote_ratio'] = pd.to_numeric(self.salary_data['remote_ratio'])

        # Drop NaN values
        self.salary_data.dropna(inplace=True)

        # Define features for encoding
        self.features = ['experience_level', 'employment_type', 'job_title', 'salary_currency', 'salary_in_usd', 'employee_residence', 'remote_ratio', 'company_location', 'company_size']
        
        # Fit OneHotEncoder
        self.encoder.fit(self.salary_data[self.features])

    def _train(self):
        # Train the model
        X = self.salary_data[['work_year'] + self.features[1:]]
        y = self.salary_data['salary']
        self.model = LogisticRegression()
        self.model.fit(X, y)

    def predict(self, data):
        try:
            # Convert relevant fields to numeric
            work_year = pd.to_numeric(data['work_year'])
            salary_in_usd = pd.to_numeric(data['salary_in_usd'])
            remote_ratio = pd.to_numeric(data['remote_ratio'])

            # Map experience level
            experience_level_mapping = {
                'entry': 'EN',
                'mid': 'MI',
                'senior': 'SE',
                'expert': 'EX'
            }
            experience_level = experience_level_mapping[data['experience_level'].lower()]

            # Map job title
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
            job_title = job_title_mapping[data['job_title'].lower()]


            input_data = pd.DataFrame([[work_year, experience_level, job_title, salary_in_usd, remote_ratio]], columns=['work_year'] + self.features[1:])
            
            # Transform categorical features
            input_data[self.features] = self.encoder.transform(input_data[self.features])

            # Predict probability
            salary_probability = self.model.predict_proba(input_data)[:, 1]
            return float(salary_probability)
        except ValueError as e:
            return {'error': str(e)}, 400

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
            parser.add_argument('employment_type', type=int, required=True)               
            parser.add_argument('job_title', type=str, required=True)
            parser.add_argument('salary_currency', type=str, required=True)
            parser.add_argument('salary_in_usd', type=int, required=True)
            parser.add_argument('employee_residence', type=str, required=True)
            parser.add_argument('remote_ratio', type=int, required=True)
            parser.add_argument('company_location', type=str, required=True)
            parser.add_argument('company_size', type=str, required=True)
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