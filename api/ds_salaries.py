from flask import Flask, Blueprint
from flask_restful import Api, Resource
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
            
            # Preprocessing
            salary_data['employment_type'] = 'ft'  # Assuming 'ft' for full-time as the only option
            salary_data['currency'] = 'USD'  # Assuming USD as the only currency
            salary_data['usd_salary'] = salary_data['salary']  # Assuming 'salary' column is in USD
            salary_data.drop(['salary'], axis=1, inplace=True)
            
            # Train logistic regression model
            X = salary_data.drop('usd_salary', axis=1)
            y = salary_data['usd_salary']
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