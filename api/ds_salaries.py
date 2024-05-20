import sys
<<<<<<< HEAD

sys.path.append('/home/sumi/vscode/flask-portfolio-1')

=======
sys.path.append('/home/sumi/vscode/flask-portfolio-1')
>>>>>>> 60edfde1c777b01b790d6b3b08932580343754da
from flask import Flask, Blueprint, request, jsonify
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
from model.salaries_model import SalaryModel

app = Flask(__name__)
salaries_api = Blueprint('api', __name__, url_prefix='/api')
api = Api(salaries_api)
CORS(salaries_api)

class SalaryAPI:
    class _Predict(Resource):
        def post(self):
            try:
                parser = reqparse.RequestParser()
<<<<<<< HEAD
                parser.add_argument('work_year', type=int, required=True)
=======
                parser.add_argument('work_year', type=str, required=True)
>>>>>>> 60edfde1c777b01b790d6b3b08932580343754da
                parser.add_argument('experience_level', type=str, required=True)
                parser.add_argument('employment_type', type=str, required=True)
                parser.add_argument('job_title', type=str, required=True)
                parser.add_argument('salary_currency', type=str, required=True)
<<<<<<< HEAD
                parser.add_argument('salary_in_usd', type=float, required=True)
=======
                parser.add_argument('salary_in_usd', type=float, required=True) 
>>>>>>> 60edfde1c777b01b790d6b3b08932580343754da
                parser.add_argument('employee_residence', type=str, required=True)
                parser.add_argument('remote_ratio', type=float, required=True)
                parser.add_argument('company_location', type=str, required=True)
                parser.add_argument('company_size', type=str, required=True)
                args = parser.parse_args()

                print("Received data:", args)

                experience_level_mapping = {
<<<<<<< HEAD
=======
                    'en': 'EN',
                    'mi': 'MI',
                    'se': 'SE',
                    'ex': 'EX',
>>>>>>> 60edfde1c777b01b790d6b3b08932580343754da
                    'entry': 'EN',
                    'mid': 'MI',
                    'senior': 'SE',
                    'expert': 'EX'
                }
<<<<<<< HEAD
                received_experience_level = args['experience_level'].lower()
=======
                received_experience_level = args['experience_level'].strip().lower()
>>>>>>> 60edfde1c777b01b790d6b3b08932580343754da
                mapped_experience_level = experience_level_mapping.get(received_experience_level)

                print("Mapped experience level:", mapped_experience_level)

                if mapped_experience_level is None:
                    raise ValueError("Invalid experience level")

                job_title_mapping = {
                    'data scientist': 1,
                    'data analyst': 2,
                    'data engineer': 3,
                    'machine learning scientist': 4,
                    'big data engineer': 5,
                    'product data analyst': 6,
                    'machine learning engineer': 7,
                    'lead data scientist': 8,
                    'business data analyst': 9,
                    'lead data engineer': 10,
                    'lead data analyst': 11,
                    'data scientist consultant': 12,
                    'bi data analyst': 13,
                    'director of data science': 14,
                    'research scientist': 15,
                    'machine learning manager': 16,
                    'data engineering manager': 17,
                    'machine learning infrastructure engineer': 18,
                    'ml engineer': 19,
                    'ai scientist': 20,
                    'computer vision engineer': 21,
                    'principal data scientist': 22,
                    'head of data': 23,
                    '3d computer vision researcher': 24,
                    'applied data scientist': 25,
                    'marketing data analyst': 26,
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
<<<<<<< HEAD
                job_title = args['job_title'].lower()
                job_title_id = job_title_mapping.get(job_title)

                if job_title_id is None:
                    raise ValueError("Invalid job title")

                salary_data = {
                    'work_year': args['work_year'],
                    'experience_level': mapped_experience_level,
                    'employment_type': args['employment_type'],
                    'job_title': job_title_id,
                    'salary_currency': args['salary_currency'],
                    'salary_in_usd': args['salary_in_usd'],
                    'employee_residence': args['employee_residence'],
                    'remote_ratio': args['remote_ratio'],
                    'company_location': args['company_location'],
                    'company_size': args['company_size']
                }

                salary_model_instance = SalaryModel.get_instance()
                response, status_code = salary_model_instance.predict_salary(salary_data)

                return jsonify(response), status_code
            except Exception as e:
                return {'error': str(e)}, 400

    api.add_resource(_Predict, '/salaries/predict')
=======
                job_title = args['job_title'].strip().lower()
                job_title_id = job_title_mapping.get(job_title)

                print("Mapped job title:", job_title_id)

                if job_title_id is None:
                    raise ValueError("Invalid job title")

                salary_data = {
                    'work_year': args['work_year'],
                    'experience_level': mapped_experience_level,
                    'employment_type': args['employment_type'],
                    'job_title': job_title_id,
                    'salary_currency': args['salary_currency'],
                    'salary_in_usd': args['salary_in_usd'],
                    'employee_residence': args['employee_residence'],
                    'remote_ratio': args['remote_ratio'],
                    'company_location': args['company_location'],
                    'company_size': args['company_size']
                }

                salary_model_instance = SalaryModel.get_instance()
                response, status_code = salary_model_instance.predict_salary(salary_data)

                return jsonify(response), status_code
            except Exception as e:
                return {'error': str(e)}, 400
>>>>>>> 60edfde1c777b01b790d6b3b08932580343754da
