from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource

salaries_api = Blueprint('salaries_api', __name__, url_prefix='/api/salaries')
api = Api(salaries_api)

class SalaryAPI:
    class _Predict(Resource):
        
        def post(self):
            salary_data = request.get_json()

            # Get the singleton instance of the SalaryModel
            salary_model = SalaryAPI.get_instance()
            # Predict the salary probability
            salary_prob = salary_model.predict(salary_data)

            # Return the response as JSON
            return jsonify({'salary_probability': salary_prob})

    api.add_resource(_Predict, '/predict')