from flask import Flask, Blueprint
from flask_restful import Api, Resource, reqparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

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
        self.encoder.fit(self.salary_data[['experience_level', 'employment_type', 'currency', 'employee_residence', 'company_location']])

    def _train(self):
        # Train the model
        X = self.salary_data[['experience_level', 'employment_type', 'currency', 'employee_residence', 'company_location']]
        y = self.salary_data['usd_salary']
        X_encoded = self.encoder.transform(X)
        self.model = LogisticRegression()
        self.model.fit(X_encoded, y)

    def predict(self, data):
        # Predict salary
        input_data = pd.DataFrame([data], columns=['experience_level', 'employment_type', 'currency', 'employee_residence', 'company_location'])
        input_data_encoded = self.encoder.transform(input_data)
        salary_prediction = self.model.predict(input_data_encoded)
        return float(salary_prediction)

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
            parser.add_argument('experience_level', type=str, required=True)
            parser.add_argument('employment_type', type=str, required=True)
            parser.add_argument('currency', type=str, required=True)
            parser.add_argument('employee_residence', type=str, required=True)
            parser.add_argument('company_location', type=str, required=True)
            args = parser.parse_args()

            # Get singleton instance of SalaryModel
            salary_model = SalaryModel.get_instance()

            # Predict salary
            salary_prediction = salary_model.predict(args)

            return {'predicted_salary': salary_prediction}
        except Exception as e:
            return {'error': str(e)}, 400

api.add_resource(Predict, '/predict')

if __name__ == "__main__":
    app.register_blueprint(salaries_api)
    app.run(debug=True)