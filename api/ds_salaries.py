from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource, reqparse
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

salaries_api = Blueprint('salaries_api', __name__, url_prefix='/api/salaries')
api = Api(salaries_api)

class SalaryModel:
    _instance = None

    def __init__(self):
        self.model = None
        self.encoder = OneHotEncoder(handle_unknown='ignore')

    def _clean(self):
        # Load and preprocess data
        csv_file_path  = ('/home/sumi/vscode/flask-portfolio-1/ds_salaries.csv')
        self.salary_data = pd.read_csv(csv_file_path)
        self.salary_data['work_year'] = pd.to_numeric(self.salary_data['work_year'])
        self.salary_data['salary'] = pd.to_numeric(self.salary_data['salary'])
        self.salary_data['salary_in_usd'] = pd.to_numeric(self.salary_data['salary_in_usd'])
        self.salary_data['remote_ratio'] = pd.to_numeric(self.salary_data['remote_ratio'])

        #Converting fields to category dtype for efficient memory usage and better performance for certain operations
        #self.salary_data['experience_level'] = self.salary_data['experience_level'].astype('category')
        #self.salary_data['employment_type'] = self.salary_data['employment_type'].astype('category')
        #self.salary_data['job_title'] = self.salary_data['job_title'].astype('category')
        #self.salary_data['salary_currency'] = self.salary_data['salary_currency'].astype('category')
        #self.salary_data['employee_residence'] = self.salary_data['employee_residence'].astype('category')
        #self.salary_data['company_location'] = self.salary_data['company_location'].astype('category')
        #self.salary_data['company_size'] = self.salary_data['company_size'].astype('category')

        self.salary_data.dropna(inplace=True)
        
        self.features = ['experience_level', 'employment_type', 'job_title', 'salary_currency', 'salary_in_usd', 'employee_residence', 'remote_ratio', 'company_location', 'company_size']
        
        self.encoder.fit(self.salary_data[self.features])
        
    def _train(self):
        # Train the model
        X = self.salary_data[['work_year'] + self.features[1:]]
        y = self.salary_data['salary']
        self.model = LogisticRegression()
        self.model.fit(X, y)

    def predict(self, data):
        try:
        # Predict salary probability
            work_year = float(data['work_year'])
            salary_in_usd = float(data['salary_in_usd'])
            remote_ratio = float(data['remote_ratio'])
        
            experience_level_mapping = {
                'entry': 'EN',
                'mid': 'MI',
                'senior': 'SE',
                'expert': 'EX'     
            }
            experience_level = experience_level_mapping.get[data['experience_level'].lower()]
            if experience_level is None:
                raise ValueError("Invalid experience level")
        
        # Map experience level to numerical values recognized by the model
        #if experience_level == 'entry':
        #    experience_level = 'EN'  # Map 'Entry Level' to 'EN'
        #elif experience_level == 'mid':
        #    experience_level = 'MI'  # Map 'Mid Level' to 'MI'
        #elif experience_level == 'senior':
        #    experience_level = 'SE'
        #elif experience_level == 'expert':
        #    experience_level = 'EX'  # Map 'Expert Level' to 'EX'
        #else:
        #    raise ValueError("Invalid experience level")
            
            job_title_mapping = {
            'Data Scientist': 1,
            'Data Analyst': 2,
            'Data Engineer': 3,
            'Machine Learning Scientist': 4,
            'Big Data Engineer': 5,
            'Product Data Analyst': 6,
            'Machine Learning Engineer': 7,
            'Lead Data Scientist': 8,
            'Business Data Analyst': 9,
            'Lead Data Engineer': 10,
            'Lead Data Analyst': 11,
            'Data Scientist Consultant': 12,
            'BI Data Analyst': 13,
            'Director of Data Science': 14,
            'Research Scientist': 15,
            'Machine Learning Manager': 16,
            'Data Engineering Manager': 17,
            'Machine Learning Infrastructure Engineer': 18,
            'ML Engineer': 19,
            'AI Scientist': 20,
            'Computer Vision Engineer': 21,
            'Principal Data Scientist': 22,
            'Head of Data': 23,
            '3D Computer Vision Researcher': 24,
            'Applied Data Scientist': 25,
            'Marketing Data Analyst': 26,
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
            job_title = job_title_mapping.get[data['job_title'].lower()]
            if job_title is None:
                raise ValueError("Invalid job title")
        
            input_data = pd.DataFrame([[work_year, experience_level, job_title, salary_in_usd, remote_ratio]], columns=['work_year'] + self.features[1:])
        
            input_data[self.features] = self.encoder.transform(input_data[self.features])
        
            salary_probability = self.model.predict_proba(input_data)[:, 1]
            return float(salary_probability)
        except (ValueError, KeyError) as e:
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

            # Get singleton instance of SalaryModel
            salary_model = SalaryModel.get_instance()

            # Predict salary
            salary_prob = salary_model.predict(args)

            return {'salary_probability': salary_prob}
        except Exception as e:
            return {'error': str(e)}, 400

api.add_resource(Predict, '/predict')