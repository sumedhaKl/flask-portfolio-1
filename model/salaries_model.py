from flask import Flask, Blueprint
from flask_restful import Api, Resource, reqparse
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
salaries_api = Blueprint('salaries_api', __name__, url_prefix='/api/salaries')
api = Api(salaries_api)

class SalaryModel:
    _instance = None

    def __init__(self):
        self.model = None
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.salary_data = pd.read_csv("ds_salaries.csv")
        
        self.features = ['work_year', 'salary_in_usd', 'remote_ratio']
        self.target = 'salary'

    def _clean(self):
        # Load and preprocess data
        #self.salary_data['work_year'] = pd.to_numeric(self.salary_data['work_year'])
        # self.salary_data['salary'] = pd.to_numeric(self.salary_data['salary'])
        # self.salary_data['salary_in_usd'] = pd.to_numeric(self.salary_data['salary_in_usd'])
        # self.salary_data['remote_ratio'] = pd.to_numeric(self.salary_data['remote_ratio'])

        onehot = self.encoder.fit_transform(self.salary_data[['experience_level']]).toarray()
        cols = ['experience_level_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        self.salary_data = pd.concat([self.salary_data, onehot_df], axis=1)
        self.salary_data.drop(['experience_level'], axis=1, inplace=True)
        
        self.features.extend(cols)
        
        onehot = self.encoder.fit_transform(self.salary_data[['employment_type']]).toarray()
        cols = ['employment_type_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        self.salary_data = pd.concat([self.salary_data, onehot_df], axis=1)
        self.salary_data.drop(['employment_type'], axis=1, inplace=True)
        
        self.features.extend(cols)


        onehot = self.encoder.fit_transform(self.salary_data[['salary_currency']]).toarray()
        cols = ['salary_currency_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        self.salary_data = pd.concat([self.salary_data, onehot_df], axis=1)
        self.salary_data.drop(['salary_currency'], axis=1, inplace=True)
        
        self.features.extend(cols)

        onehot = self.encoder.fit_transform(self.salary_data[['employee_residence']]).toarray()
        cols = ['employee_residence_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        self.salary_data = pd.concat([self.salary_data, onehot_df], axis=1)
        self.salary_data.drop(['employee_residence'], axis=1, inplace=True)
        
        self.features.extend(cols)


        onehot = self.encoder.fit_transform(self.salary_data[['job_title']]).toarray()
        cols = ['job_title_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        self.salary_data = pd.concat([self.salary_data, onehot_df], axis=1)
        self.salary_data.drop(['job_title'], axis=1, inplace=True)
        
        self.features.extend(cols)

        onehot = self.encoder.fit_transform(self.salary_data[['company_location']]).toarray()
        cols = ['company_location_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        self.salary_data = pd.concat([self.salary_data, onehot_df], axis=1)
        self.salary_data.drop(['company_location'], axis=1, inplace=True)
        
        self.features.extend(cols)
        
        onehot = self.encoder.fit_transform(self.salary_data[['company_size']]).toarray()
        cols = ['company_size_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        self.salary_data = pd.concat([self.salary_data, onehot_df], axis=1)
        self.salary_data.drop(['company_size'], axis=1, inplace=True)
        
        self.features.extend(cols)

        #Converting fields to category dtype for efficient memory usage and better performance for certain operations
        #self.salary_data['experience_level'] = self.salary_data['experience_level'].astype('float')
        #self.salary_data['employment_type'] = self.salary_data['employment_type'].astype('category')
        #self.salary_data['job_title'] = self.salary_data['job_title'].astype('category')
        #self.salary_data['salary_currency'] = self.salary_data['salary_currency'].astype('category')
        #self.salary_data['employee_residence'] = self.salary_data['employee_residence'].astype('category')
        #self.salary_data['company_location'] = self.salary_data['company_location'].astype('category')
        #self.salary_data['company_size'] = self.salary_data['company_size'].astype('category')

        #self.encoder.fit(self.salary_data[['work_year', 'experience_level', 'employment_type', 'job_title', 'salary_currency', 'salary_in_usd', 'employee_residence', 'remote_ratio', 'company_location', 'company_size']])

    def _train(self):
        # Train the model
        X = self.salary_data[self.features]
        y = self.salary_data[self.target]
        self.model = LogisticRegression()
        self.model.fit(X, y)

    def predict(self, data):
        # Predict salary probability
        work_year = data['work_year']
        experience_level = data['experience_level']
        employment_type = data['employment_type']
        job_title = data['job_title']
        salary_currency = data['salary_currency']
        salary_in_usd = data['salary_in_usd']
        employee_residence = data['employee_residence']
        remote_ratio = data['remote_ratio']
        company_location = data['company_location']
        company_size = data['company_size']
        
        # Map experience level to numerical values recognized by the model
        #if experience_level == 'entry':
        #    experience_level = 'EN'  # Map 'Entry Level' to 'EN'
        #elif experience_level == 'mid':
        #    experience_level = 'MI'  # Map 'Mid Level' to 'MI'
        #elif experience_level == 'senior':
        #    experience_level = 'SE'
        #elif experience_level == 'expert':
        #    experience_level = 'EX'  # Map 'Expert Level' to 'EX'
        #else:
        #    raise ValueError("Invalid experience level")
        
        salary_df = pd.DataFrame(data, index=[0])
        onehot = self.encoder.fit_transform(salary_df[['experience_level']]).toarray()
        cols = ['experience_level_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        salary_df = pd.concat([salary_df, onehot_df], axis=1)
        salary_df.drop(['experience_level'], axis=1, inplace=True)

        self.features.extend(cols)
        onehot = self.encoder.fit_transform(self.salary_data[['employment_type']]).toarray()
        cols = ['employment_type_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        self.salary_data = pd.concat([self.salary_data, onehot_df], axis=1)
        self.salary_data.drop(['employment_type'], axis=1, inplace=True)

        onehot = self.encoder.fit_transform(self.salary_data[['salary_currency']]).toarray()
        cols = ['salary_currency_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        self.salary_data = pd.concat([self.salary_data, onehot_df], axis=1)
        self.salary_data.drop(['salary_currency'], axis=1, inplace=True)

        onehot = self.encoder.fit_transform(self.salary_data[['employee_residence']]).toarray()
        cols = ['employee_residence_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        self.salary_data = pd.concat([self.salary_data, onehot_df], axis=1)
        self.salary_data.drop(['employee_residence'], axis=1, inplace=True)
        
        onehot = self.encoder.fit_transform(salary_df[['job_title']]).toarray()
        cols = ['job_title_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        salary_df = pd.concat([salary_df, onehot_df], axis=1)
        salary_df.drop(['job_title'], axis=1, inplace=True)
        
        onehot = self.encoder.fit_transform(self.salary_data[['company_location']]).toarray()
        cols = ['company_location_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        self.salary_data = pd.concat([self.salary_data, onehot_df], axis=1)
        self.salary_data.drop(['company_location'], axis=1, inplace=True)
        
        onehot = self.encoder.fit_transform(self.salary_data[['company_size']]).toarray()
        cols = ['company_size_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        self.salary_data = pd.concat([self.salary_data, onehot_df], axis=1)
        self.salary_data.drop(['company_size'], axis=1, inplace=True)
        
        # if job_title == 'Data Scientist':
        #     job_title = 1  
        # elif job_title == 'Data Analyst':
        #     job_title = 2
        # elif job_title == 'Data Engineer':
        #     job_title = 3
        # elif job_title == 'Machine Learning Scientist':
        #     job_title = 4
        # elif job_title == 'Big Data Engineer':
        #     job_title = 5
        # elif job_title == 'Product Data Analyst':
        #     job_title = 6
        # elif job_title == 'Machine Learning Engineer':
        #     job_title = 7
        # elif job_title == 'Lead Data Scientist':
        #     job_title = 8
        # elif job_title == 'Business Data Analyst':
        #     job_title = 9
        # elif job_title == 'Lead Data Engineer':
        #     job_title = 10
        # elif job_title == 'Lead Data Analyst':
        #     job_title = 11
        # elif job_title == 'Data Scientist Consultant':
        #     job_title = 12
        # elif job_title == 'BI Data Analyst':
        #     job_title = 13
        # elif job_title == 'Director of Data Science':
        #     job_title = 14
        # elif job_title == 'Research Scientist':
        #     job_title = 15
        # elif job_title == 'Machine Learning Manager':
        #     job_title = 16
        # elif job_title == 'Data Engineering Manager':
        #     job_title = 17
        # elif job_title == 'Machine Learning Infrastructure Engineer':
        #     job_title = 18
        # elif job_title == 'ML Engineer':
        #     job_title = 19
        # elif job_title == 'AI Scientist':
        #     job_title = 20
        # elif job_title == 'Computer Vision Engineer':
        #     job_title = 21
        # elif job_title == 'Principal Data Scientist':
        #     job_title = 22
        # elif job_title == 'Head of Data':
        #     job_title = 23
        # elif job_title == '3D Computer Vision Researcher':
        #     job_title = 24
        # elif job_title == 'Applied Data Scientist':
        #     job_title = 25
        # elif job_title == 'Marketing Data Analyst':
        #     job_title = 26
        # elif job_title == 'Cloud Data Engineer':
        #     job_title = 27
        # elif job_title == 'Financial Data Analyst':
        #     job_title = 28
        # elif job_title == 'Computer Vision Software Engineer':
        #     job_title = 29
        # elif job_title == 'Data Science Manager':
        #     job_title = 30
        # elif job_title == 'Data Analytics Engineer':
        #     job_title = 31
        # elif job_title == 'Applied Machine Learning Scientist':
        #     job_title = 32
        # elif job_title == 'Data Specialist':
        #     job_title = 33
        # elif job_title == 'Data Science Engineer':
        #     job_title = 34
        # elif job_title == 'Big Data Architect':
        #     job_title = 35
        # elif job_title == 'Head of Data Science':
        #     job_title = 36
        # elif job_title == 'Analytics Engineer':
        #     job_title = 37
        # elif job_title == 'Data Architect':
        #     job_title = 38
        # elif job_title == 'Head of Machine Learning':
        #     job_title = 39
        # elif job_title == 'ETL Developer':
        #     job_title = 40
        # elif job_title == 'Lead Machine Learning Engineer':
        #     job_title = 41
        # elif job_title == 'Machine Learning Developer':
        #     job_title = 42
        # elif job_title == 'Principal Data Analyst':
        #     job_title = 43
        # elif job_title == 'Machine Learning Infrastructure Engineer':
        #     job_title = 44
        # elif job_title == 'NLP Engineer':
        #     job_title = 45
        # elif job_title == 'Data Analytics Lead':
        #     job_title = 46
        # else:
        #     raise ValueError("Invalid job title")
        
        input_data = pd.DataFrame([[work_year, experience_level, employment_type, job_title, salary_currency, salary_in_usd, employee_residence, remote_ratio, company_location, company_size]], columns=['work_year', 'experience_level', 'employment_type', 'job_title', 'salary_currency', 'salary_in_usd', 'employee_residence', 'remote_ratio', 'company_location', 'company_size'])
        input_data[['work_year', 'experience_level', 'employment_type', 'job_title', 'salary_currency', 'salary_in_usd', 'employee_residence', 'remote_ratio', 'company_location', 'company_size']] = self.encoder.transform(input_data[['work_year', 'experience_level', 'employment_type', 'job_title', 'salary_currency', 'salary_in_usd', 'employee_residence', 'remote_ratio', 'company_location', 'company_size']])
        salary_probability = self.model.predict_proba(input_data)[:, 1]
        return float(salary_probability)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._clean()
            cls._instance._train()
        return cls._instance

def initSalary():
    SalaryModel.get_instance()

class Predict(Resource):
    def post(self):
        try:
            # Parse incoming request data
            parser = reqparse.RequestParser()
            parser.add_argument('work_year', type=int, required=True)
            parser.add_argument('experience_level', type=str, required=True)
            parser.add_argument('employment_type', type=int, required=True)               
            parser.add_argument('job_title', type=str, required=True)
            parser.add_argument('salary_currency', type=str, required=True)
            parser.add_argument('salary_in_usd', type=float, required=True)
            parser.add_argument('employee_residence', type=str, required=True)
            parser.add_argument('remote_ratio', type=float, required=True)
            parser.add_argument('company_location', type=str, required=True)
            parser.add_argument('company_size', type=str, required=True)
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