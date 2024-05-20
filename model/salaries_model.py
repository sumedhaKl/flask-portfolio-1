from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class SalaryModel:
    _instance = None

    def __init__(self):
        self.model = None
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.features = ['work_year', 'experience_level', 'employment_type', 'salary_in_usd', 'remote_ratio']
        self.target = 'salary'

        csv_file_path = "/home/kasm-user/vscode/flask-portfolio-1/ds_salaries.csv"
        self.salary_data = pd.read_csv(csv_file_path)
        
        self._clean()
        self._train()

    def _clean(self):
        self.salary_data.dropna(inplace=True)

    def _train(self):
        X = self.salary_data[self.features]
        y = self.salary_data[self.target]
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)

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