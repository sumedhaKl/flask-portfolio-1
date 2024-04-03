from flask import Flask, Blueprint
from flask_restful import Api, Resource
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import os

app = Flask(__name__)
salaries_api = Blueprint('salaries_api', __name__, url_prefix='/api/salaries')
api = Api(salaries_api)

class SalaryPredictor(Resource):
    def post(self):
        try:
            # Get the directory of the script
            script_dir = os.path.dirname(__file__)
            
            # Define the path to the CSV file
            csv_file_path = os.path.join(script_dir, 'ds_salaries.csv')
            
            # Load salary data
            salary_data = pd.read_csv(csv_file_path)
            
            encoder = OneHotEncoder(handle_unknown='ignore')
            encoder.fit(salary_data[['work_year', 'experience_level', 'employment_type', 'job_title', 'salary_currency', 'salary_in_usd', 'employee_residence', 'remote_ratio', 'company_location', 'company_size']])
           
            # Preprocessing
            encoder=OneHotEncoder(handle_unknown='ignore')
            salary_data['experience_level'] = salary_data['experience_level'].map({'entry': 'EN', 'mid': 'MI', 'senior': 'SE', 'expert': 'EX'})
            salary_data['employment_type'] = salary_data['employment_type'].map({'ft': 1})  # Assuming 'ft' for full-time as the only option
            salary_data['salary_currency'] = salary_data['salary_currency'].map({'USD': 1})  # Assuming USD as the only currency
            salary_data['salary_in_usd'] = salary_data['salary_in_usd']  # Assuming 'salary' column is in USD
            
            salary_data = salary_data[['work_year', 'experience_level', 'employment_type', 'job_title', 'salary_currency', 'salary_in_usd', 'employee_residence', 'remote_ratio', 'company_location', 'company_size']]
            
            # Train logistic regression model
            X = salary_data.drop('salary_in_usd', axis=1)
            y = salary_data['salary_in_usd']
            logreg = LogisticRegression()
            logreg.fit(X, y)
            
            # Predict salary probabilities
            salary_prob = logreg.predict_proba(X)[:, 1]

            return {'salary_probabilities': list(map(float, salary_prob))}
        except Exception as e:
            return {'error': str(e)}, 400

api.add_resource(SalaryPredictor, '/predict')

if __name__ == "__main__":
    app.register_blueprint(salaries_api)
    app.run(debug=True)