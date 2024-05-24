import sys
sys.path.append('/home/sumi/vscode/flask-portfolio-1')
from flask import Blueprint, Flask, request, jsonify
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
import pandas as pd
from model.salaries_model import SalaryModel

app = Flask(__name__)
salaries_api = Blueprint('salaries_api', __name__, url_prefix='/api/salaries')
api = Api(salaries_api)
CORS(salaries_api)

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

            salary_model = SalaryModel.get_instance()
            response, status_code = salary_model.predict(args)
            return response, status_code
        except Exception as e:
            return {'error': str(e)}, 400

api.add_resource(Predict, '/predict')

if __name__ == "__main__":
    app.register_blueprint(salaries_api)
    app.run(debug=True)