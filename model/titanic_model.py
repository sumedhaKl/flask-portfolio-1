import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from __init__ import db  # Definitions initialization

class predict(db.Model):
    def Predict(f_data):
        # Logistic regression model is used to predict the probability

        if type(f_data) != type(pd.DataFrame({})):
            return -1

        passenger = f_data

        new_passenger = passenger.copy()

            # Preprocess the new passenger data
        new_passenger['sex'] = new_passenger['sex'].apply(lambda x: 1 if x == 'male' else 0)
        new_passenger['alone'] = new_passenger['alone'].apply(lambda x: 1 if x == True else 0)

            # Encode 'embarked' variable
        enc = OneHotEncoder(handle_unknown='ignore')
        onehot = enc.transform(new_passenger[['embarked']]).toarray()
        cols = ['embarked_' + val for val in enc.categories_[0]]
        new_passenger[cols] = pd.DataFrame(onehot, index=new_passenger.index)
        new_passenger.drop(['name'], axis=1, inplace=True)
        new_passenger.drop(['embarked'], axis=1, inplace=True)

        # Split arrays in random train 70%, random test 30%, using stratified sampling (same proportion of survived in both sets) and a fixed random state (42
        # The number 42 is often used in examples and tutorials because of its cultural significance in fields like science fiction (it's the "Answer to the Ultimate Question of Life, The Universe, and Everything" in The Hitchhiker's Guide to the Galaxy by Douglas Adams). But in practice, the actual value doesn't matter; what's important is that it's set to a consistent value.
        # X_train is the DataFrame containing the features for the training set.
        # X_test is the DataFrame containing the features for the test set.
        # y-train is the 'survived' status for each passenger in the training set, corresponding to the X_train data.
        # y_test is the 'survived' status for each passenger in the test set, corresponding to the X_test data.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train a logistic regression model
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)

        return logreg.predict(new_passenger)

    def __init__(self):
        # the titanic ML model
        self.model = None
        self.dt = None
        # define ML features and target
        self.features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'alone']
        self.target = 'survived'
        # load the titanic dataset
        self.titanic_data = sns.load_dataset('titanic')
        # one-hot encoder used to encode 'embarked' column
        self.encoder = OneHotEncoder(handle_unknown='ignore')

        def _train(logreg):
                # split the data into features and target
                X = self.titanic_data[self.features]
                y = self.titanic_data[self.target]
                
                # perform train-test split
                self.model = LogisticRegression(max_iter=1000)
                
                # train the model
                self.model.fit(X, y)
                
                # train a decision tree classifier
                self.dt.fit(X, y)
            
        # Predict the survival probability for the new passenger
    dead_proba, alive_proba = np.squeeze(logreg.predict_proba(new_passenger))
    
def initTitanicData():
      # Load the titanic dataset
    titanic_data = sns.load_dataset('titanic')
    
    td = titanic_data
    td.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'], axis=1, inplace=True)
    td.dropna(inplace=True) # drop rows with at least one missing value, after dropping unuseful columns
    td['sex'] = td['sex'].apply(lambda x: 1 if x == 'male' else 0)
    td['alone'] = td['alone'].apply(lambda x: 1 if x is True else 0)

    # Encode categorical variables
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(td[['embarked']])
    onehot = enc.transform(td[['embarked']]).toarray()
    cols = ['embarked_' + val for val in enc.categories_[0]]
    td[cols] = pd.DataFrame(onehot)
    td.drop(['embarked'], axis=1, inplace=True)
    td.dropna(inplace=True) # drop rows with at least one missing value, after preparing the data
# Build distinct data frames on survived column
X = ('survived') # all except 'survived'
y = ['survived'] # only 'survived'