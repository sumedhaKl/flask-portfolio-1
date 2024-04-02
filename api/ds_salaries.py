from flask import Flask, Blueprint
from flask_restful import Api, Resource
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
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
            salary_data['job_title'] = salary_data['job_title'].apply(lambda x: 1 if x else 0)
            salary_data['experience_level'] = salary_data['experience_level'].apply(lambda x: 1 if x else 0)
            
            # One-hot encode salary data
            enc = OneHotEncoder(handle_unknown='ignore')
            onehot = enc.fit_transform(salary_data[['job_title', 'experience_level']]).toarray()
            cols = ['job_title_' + str(val) for val in enc.categories_[0]] + ['experience_level_' + str(val) for val in enc.categories_[1]]
            salary_data[cols] = pd.DataFrame(onehot)
            salary_data.drop(['job_title', 'experience_level'], axis=1, inplace=True)
            
            # Train logistic regression model
            X = salary_data.drop('salary', axis=1)
            y = salary_data['salary']
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