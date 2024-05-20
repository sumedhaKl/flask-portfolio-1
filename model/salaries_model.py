<<<<<<< HEAD
=======
from flask import Blueprint, Flask
from flask_restful import Api, reqparse
>>>>>>> 60edfde1c777b01b790d6b3b08932580343754da
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

<<<<<<< HEAD
class SalaryModel:
=======
app = Flask(__name__)
salaries_api = Blueprint('salaries_api', __name__, url_prefix='/api/salaries')
api = Api(salaries_api)

csv_file_path = "/home/sumi/vscode/flask-portfolio-1/ds_salaries.csv"
df = pd.read_csv(csv_file_path)        

class SalaryModel:
    
>>>>>>> 60edfde1c777b01b790d6b3b08932580343754da
    _instance = None

    def __init__(self):
        self.model = None
<<<<<<< HEAD
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.features = ['work_year', 'experience_level', 'employment_type', 'salary_in_usd', 'remote_ratio']
        self.target = 'salary'

        csv_file_path = "/home/kasm-user/vscode/flask-portfolio-1/ds_salaries.csv"
        self.salary_data = pd.read_csv(csv_file_path)
=======
        self.features = ['work_year','salary_in_usd', 'remote_ratio']
        self.target = 'salary'
        
        encoder = OneHotEncoder(handle_unknown='ignore')
        self.salary_data = df
        self.encoder = encoder 
>>>>>>> 60edfde1c777b01b790d6b3b08932580343754da
        
        self._clean()
        self._train()

    def _clean(self):
        self.salary_data.dropna(inplace=True)

    def _train(self):
        X = self.salary_data[self.features]
        y = self.salary_data[self.target]
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)
<<<<<<< HEAD

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def predict_salary(self, input_data):
        try:
            input_features = pd.DataFrame([input_data], columns=self.features)
            input_features_encoded = self.encoder.transform(input_features[['experience_level', 'employment_type']]).toarray()
            input_features.drop(['experience_level', 'employment_type'], axis=1, inplace=True)
            input_features = pd.concat([input_features, pd.DataFrame(input_features_encoded)], axis=1)

            probability = self.model.predict_proba(input_features)[0, 1]
            return {'probability': probability}, 200
        except Exception as e:
            return {'error': str(e)}, 400
=======
        
    def predict_salary(self, input_data):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('work_year', type=float, required=True) 
            parser.add_argument('salary_in_usd', type=float, required=True)
            parser.add_argument('remote_ratio', type=float, required=True)
            args = parser.parse_args()
            
            input_data = pd.DataFrame([args], columns=self.features)
            predicted_salary = self.model.predict_proba(input_data)[0, 1]
            return {'predicted_salary': float(predicted_salary)}, 200 
        except Exception as e:
            return {'error': str(e)}, 400

api.add_resource(SalaryModel, '/predict')

if __name__ == '__main__':
    app.register_blueprint(salaries_api)
    app.run(debug=True)
>>>>>>> 60edfde1c777b01b790d6b3b08932580343754da
