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
        self.salary_data['experience_level'] = self.salary_data['experience_level'].apply(lambda x: 1 if x else 0)
        self.salary_data['employment_type'] = self.salary_data['employment_type'].apply(lambda x: 1 if x else 0)
        self.salary_data['job_title'] = self.salary_data['job_title'].apply(lambda x: 1 if x else 0)
        self.salary_data['salary_currency'] = self.salary_data['salary_currency'].apply(lambda x: 1 if x else 0)
        self.salary_data['salary_in_usd'] = self.salary_data['salary_in_usd'].apply(lambda x: 1 if x else 0)
        self.salary_data['employee_residence'] = self.salary_data['employee_residence'].apply(lambda x: 1 if x else 0)
        self.salary_data['remote_ratio'] = self.salary_data['remote_ratio'].apply(lambda x: 1 if x else 0)
        self.salary_data['company_location'] = self.salary_data['company_location'].apply(lambda x: 1 if x else 0)
        self.salary_data['company_size'] = self.salary_data['company_size'].apply(lambda x: 1 if x else 0)
        self.encoder.fit(self.salary_data[['work_year', 'experience_level', 'employment_type', 'job_title', 'salary_currency', 'salary_in_usd', 'employee_residence', 'remote_ratio', 'company_location', 'company_size']])

    def _train(self):
        # Train the model
        X = self.salary_data[['work_year', 'experience_level', 'employment_type', 'job_title', 'salary_currency', 'salary_in_usd', 'employee_residence', 'remote_ratio', 'company_location', 'company_size']]
        y = self.salary_data['salary']
        self.model = LogisticRegression()
        self.model.fit(X, y)

    def predict(self, data):
        # Predict salary probability
        work_year = data['work_year']
        experience_level = data['experience_level']
        employment_type = data['employment_type']
        job_title = data['job_title']
        salary_currency = data['salary_currency']
        salary_in_usd = data['salary_in_usd']
        employee_residence = data['employee_residence']
        remote_ratio = data['remote_ratio']
        company_location = data['company_location']
        company_size = data['company_size']
        # Map experience level to numerical values recognized by the model
        if experience_level == 'entry':
            experience_level_num = 'EN'  # Map 'Entry Level' to 'EN'
        elif experience_level == 'mid':
            experience_level_num = 'MI'  # Map 'Mid Level' to 'MI'
        elif experience_level == 'senior':
            experience_level_num = 'SE'
        elif experience_level == 'expert':
            experience_level_num = 'EX'  # Map 'Expert Level' to 'EX'
        else:
            raise ValueError("Invalid experience level")
        
        input_data = pd.DataFrame([[work_year, experience_level_num, employment_type, job_title, salary_currency, salary_in_usd, employee_residence, remote_ratio, company_location, company_size]], columns=['work_year', 'experience_level', 'employment_type', 'job_title', 'currency', 'usd_salary', 'employee_residence', 'remote_ratio', 'company_location', 'company_size'])
        input_data[['work_year', 'experience_level', 'employment_type', 'job_title', 'salary_currency', 'salary_in_usd', 'employee_residence', 'remote_ratio', 'company_location', 'company_size']] = self.encoder.transform(input_data[['work_year', 'experience_level', 'employment_type', 'job_title', 'currency', 'usd_salary', 'employee_residence', 'remote_ratio', 'company_location', 'company_size']])
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
            parser.add_argument('job_title', type=str, required=True)
            parser.add_argument('salary_currency', type=str, required=True)
            parser.add_argument('salary_in_usd', type=float, required=True)
            parser.add_argument('employee_residence', type=str, required=True)
            parser.add_argument('remote_ratio', type=float, required=True)
            parser.add_argument('company_location', type=str, required=True)
            parser.add_argument('company_size', type=int, required=True)
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