from flask import Blueprint, Flask
from flask_restful import Api, Resource
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

app = Flask(__name__)
salaries_api = Blueprint('salaries_api', __name__, url_prefix='/api/salaries')
api = Api(salaries_api)

# Load the dataset
csv_file_path = "/home/sumi/vscode/flask-portfolio-1/ds_salaries.csv"
df = pd.read_csv(csv_file_path)

class SalaryModel(Resource):
    
    _instance = None
    
    def __init__(self):
        
        self.model = None
        self.features = ['work_year', 'salary_in_usd', 'remote_ratio']
        self.target = 'salary'
        
        encoder = OneHotEncoder(handle_unknown='ignore')
        self.salary_data = df
        self.encoder = encoder
        
        self._clean()
        self._train() 
        
    def _clean(self):
        self.salary_data.dropna(inplace=True)
        
    def _train(self):
        self.model = LogisticRegression(max_iter=1000)
        
        X = self.salary_data[self.features]
        y = self.salary_data[self.target]
        
        self.model.fit(X,y)
    
    def post(self):
        try:
            input_features = pd.DataFrame([self.features], columns=self.features)
            
            probability = self.model.predict_proba(input_features)[0, 1]  # Get the probability for class 1
            return {'probability': probability}, 200
        except Exception as e:
            return {'error': str(e)}, 400

api.add_resource(SalaryModel, '/predict')

if __name__ == "__main__":
    app.register_blueprint(salaries_api)
    app.run(debug=True)