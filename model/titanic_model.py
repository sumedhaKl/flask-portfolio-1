from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import numpy as np

class TitanicModel:
    
    _instance = None
    
    def __init__(self):
        self.model = None
        self.dt = None
        self.features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'alone']
        self.target = 'survived'
        self.titanic_data = sns.load_dataset('titanic')
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        
    def _clean(self):
        self.titanic_data.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'], axis=1, inplace=True)
        self.titanic_data['sex'] = self.titanic_data['sex'].apply(lambda x: 1 if x == 'male' else 0)
        self.titanic_data['alone'] = self.titanic_data['alone'].apply(lambda x: 1 if x else 0)
        self.titanic_data.dropna(subset=['embarked'], inplace=True)
        onehot = self.encoder.fit_transform(self.titanic_data[['embarked']]).toarray()
        cols = ['embarked_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        self.titanic_data = pd.concat([self.titanic_data, onehot_df], axis=1)
        self.features.extend(cols)
        self.titanic_data.dropna(inplace=True)
        
    def _train(self):
        X = self.titanic_data[self.features]
        y = self.titanic_data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X_train, y_train)
        self.dt = DecisionTreeClassifier()
        self.dt.fit(X_train, y_train)
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._clean()
            cls._instance._train()
        return cls._instance

    def predict(self, passenger):
        passenger_df = pd.DataFrame(passenger, index=[0])
        passenger_df['sex'] = passenger_df['sex'].apply(lambda x: 1 if x == 'male' else 0)
        passenger_df['alone'] = passenger_df['alone'].apply(lambda x: 1 if x else 0)
        onehot = self.encoder.transform(passenger_df[['embarked']]).toarray()
        cols = ['embarked_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        passenger_df = pd.concat([passenger_df, onehot_df], axis=1)
        passenger_df.drop(['embarked', 'name'], axis=1, inplace=True)
        dieProbability, surviveProbability = np.squeeze(self.model.predict_proba(passenger_df))
        return {'die': dieProbability, 'survive': surviveProbability}
    
    def feature_weights(self):
        importances = self.dt.feature_importances_
        return {feature: importance for feature, importance in zip(self.features, importances)}    

def initTitanic():
    TitanicModel.get_instance()
    
if __name__ == "__main__":
    print(" Begin:", TitanicModel.__doc__)
    TitanicModel()