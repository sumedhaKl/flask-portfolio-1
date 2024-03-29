import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np

class Predict:
    def __init__(self):
        self.features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'alone']
        self.target = 'survived'
        self.model = None

    def predict(self, f_data):
        # Logistic regression model is used to predict the probability
        if not isinstance(f_data, pd.DataFrame):
            return -1

        passenger = f_data.copy()

        # Preprocess the new passenger data
        passenger['sex'] = passenger['sex'].apply(lambda x: 1 if x == 'male' else 0)
        passenger['alone'] = passenger['alone'].apply(lambda x: 1 if x else 0)

        # Encode 'embarked' variable
        enc = OneHotEncoder(handle_unknown='ignore')
        onehot = enc.transform(passenger[['embarked']]).toarray()
        cols = ['embarked_' + val for val in enc.categories_[0]]
        passenger[cols] = pd.DataFrame(onehot, index=passenger.index)
        passenger.drop(['name'], axis=1, inplace=True)
        passenger.drop(['embarked'], axis=1, inplace=True)
        
        return self.model.predict(passenger)
      
def initTitanicData():
      # Load the titanic dataset
    titanic_data = sns.load_dataset('titanic')
    
     # Define X and y
    X = titanic_data.drop('survived', axis=1)  # Features
    y = titanic_data['survived']  # Target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the model
    model = LogisticRegression(max_iter=1000)

    # Train the model
    model.fit(X_train, y_train)

    predictor = Predict()
    predictor.model = model

    return predictor

predictor = initTitanicData()