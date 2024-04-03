from flask import Flask
from flask import Blueprint
from flask_restful import Api, Resource, reqparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import pandas as pd

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
        self.salary_data['job_title'] = self.salary_data['job_title'].apply(lambda x: 1 if x else 0)
        self.salary_data['experience_level'] = self.salary_data['experience_level'].apply(lambda x: 1 if x else 0)
        self.encoder.fit(self.salary_data[['job_title', 'experience_level']])

    def _train(self):
        # Train the model
        X = self.salary_data[['job_title', 'experience_level']]
        y = self.salary_data['salary']
        self.model = LogisticRegression()
        self.model.fit(X, y)

    def predict(self, data):
        # Predict salary probability
        job_title = data['job_title']
        experience_level = data['experience_level']
        # Map experience level to numerical values recognized by the model
        if experience_level == 'EN':
            experience_level_num = 'entry'  # Map 'Entry Level' to 'EN'
        elif experience_level == 'MI':
            experience_level_num = 'mid'  # Map 'Mid Level' to 'MI'
        elif experience_level == 'SE':
            experience_level_num = 'senior'
        elif experience_level == 'EX':
            experience_level_num = 'expert'  # Map 'Expert Level' to 'EX'
        else:
            raise ValueError("Invalid experience level")
        
        input_data = pd.DataFrame([[job_title, experience_level_num]], columns=['job_title', 'experience_level'])
        input_data[['job_title', 'experience_level']] = self.encoder.transform(input_data[['job_title', 'experience_level']])
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
            parser.add_argument('job_title', type=int, required=True)
            parser.add_argument('experience_level', type=str, required=True)
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