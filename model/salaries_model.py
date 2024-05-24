import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

class SalaryModel:
    _instance = None

    def __init__(self):
        self.model = None
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.features = ['work_year', 'experience_level', 'employment_type', 'job_title', 'salary_in_usd', 'remote_ratio']
        self.target = 'salary_in_usd'
        self._clean()
        self._train()

    def _clean(self):
        csv_file_path = '/home/sumi/vscode/flask-portfolio-1/ds_salaries.csv'
        self.salary_data = pd.read_csv(csv_file_path)
        self.salary_data.dropna(inplace=True)
        self.salary_data['work_year'] = pd.to_numeric(self.salary_data['work_year'])
        self.salary_data['salary_in_usd'] = pd.to_numeric(self.salary_data['salary_in_usd'])
        self.salary_data['remote_ratio'] = pd.to_numeric(self.salary_data['remote_ratio'])
        self.encoder.fit(self.salary_data[['experience_level', 'employment_type', 'job_title']])

    def _train(self):
        X = self.salary_data[self.features].copy()
        X_encoded = self.encoder.transform(X[['experience_level', 'employment_type', 'job_title']]).toarray()
        X = X.drop(['experience_level', 'employment_type', 'job_title'], axis=1)
        X = pd.concat([X.reset_index(drop=True), pd.DataFrame(X_encoded)], axis=1)
        X.columns = X.columns.astype(str)  
        y = self.salary_data[self.target]
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)

    def predict(self, data):
        try:
            input_data = pd.DataFrame([data], columns=self.features)
            input_data_encoded = self.encoder.transform(input_data[['experience_level', 'employment_type', 'job_title']]).toarray()
            input_data = input_data.drop(['experience_level', 'employment_type', 'job_title'], axis=1)
            input_data = pd.concat([input_data.reset_index(drop=True), pd.DataFrame(input_data_encoded)], axis=1)
            input_data.columns = input_data.columns.astype(str)  
            salary_prediction = self.model.predict(input_data)[0]
            return {'predicted_salary': float(salary_prediction)}, 200
        except (ValueError, KeyError) as e:
            return {'error': str(e)}, 400

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance