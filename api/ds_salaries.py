from flask import Flask, Blueprint, request, jsonify
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS

app = Flask(__name__)
salaries_api = Blueprint('api', __name__, url_prefix='/api')
CORS(salaries_api)
api = Api(salaries_api)

class SalaryPrediction(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('work_year', type=int, required=True)
            parser.add_argument('experience_level', type=str, required=True)
            parser.add_argument('employment_type', type=str, required=True)
            parser.add_argument('job_title', type=str, required=True)
            parser.add_argument('salary_currency', type=str, required=True)
            parser.add_argument('salary_in_usd', type=float, required=True)
            parser.add_argument('employee_residence', type=str, required=True)
            parser.add_argument('remote_ratio', type=float, required=True)
            parser.add_argument('company_location', type=str, required=True)
            parser.add_argument('company_size', type=str, required=True)
            args = parser.parse_args()

            print("Received data:", args)

            received_experience_level = args['experience_level']

            print("Received experience level:", received_experience_level)

            experience_level_mapping = {
                'entry': 'EN',
                'mid': 'MI',
                'senior': 'SE',
                'expert': 'EX'     
            }

            mapped_experience_level = experience_level_mapping.get(received_experience_level.lower())

            print("Mapped experience level:", mapped_experience_level)

            if mapped_experience_level is None:
                raise ValueError("Invalid experience level")
            else:
                print("Experience level is valid:", mapped_experience_level)

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
            job_title = job_title_mapping.get(args['job_title'], None)
            if job_title is None:
                raise ValueError("Invalid job title")

            return {'success': True}, 200
        
        except Exception as e:
            return {'error': str(e)}, 400

api.add_resource(SalaryPrediction, '/salaries/predict')

if __name__ == '__main__':
    app.register_blueprint(salaries_api)
    app.run(debug=True)