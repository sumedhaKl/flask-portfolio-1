from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import seaborn as sns
import numpy as np
from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource 

 # a singleton instance of TitanicModel, created to train the model only once, while using it for prediction multiple times
_instance = None

class TitanicModel:
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
        
    def _clean(self):
        # Drop unnecessary columns
        self.titanic_data.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'], axis=1, inplace=True)

        # Convert boolean columns to integers
        self.titanic_data['sex'] = self.titanic_data['sex'].apply(lambda x: 1 if x == 'male' else 0)
        self.titanic_data['alone'] = self.titanic_data['alone'].apply(lambda x: 1 if x else 0)

        # Drop rows with missing 'embarked' values before one-hot encoding
        self.titanic_data.dropna(subset=['embarked'], inplace=True)
        
        # One-hot encode 'embarked' column
        onehot = self.encoder.fit_transform(self.titanic_data[['embarked']]).toarray()
        cols = ['embarked_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        self.titanic_data = pd.concat([self.titanic_data, onehot_df], axis=1)
        self.titanic_data.drop(['embarked'], axis=1, inplace=True)

        # Add the one-hot encoded 'embarked' features to the features list
        self.features.extend(cols)
        
        # Drop rows with missing values
        self.titanic_data.dropna(inplace=True)

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
        
        #return self.model.predict(passenger)
        die, survive = np.squeeze(self.model.predict_proba(passenger))
        
        return {'die' : die, 'survive' : survive}
    
    def _train(self):
        # split the data into features and target
        X = self.titanic_data[self.features]
        y = self.titanic_data[self.target]
        
        # perform train-test split
        self.model = LogisticRegression(max_iter=1000)
        
        # train the model
        self.model.fit(X, y)
        
        # train a decision tree classifier
        self.dt = DecisionTreeClassifier()
        self.dt.fit(X, y)
        
    @classmethod
    def get_instance(cls):

        if cls._instance is None:
            cls._instance = cls()
        cls._instance._clean()
        cls._instance._train()
        # return the instance, to be used for prediction
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

    # predict the survival probability and extract the probabilities from

    die, survive = np.squeeze(self.model.predict_proba(passenger_df))
    # return the survival probabilities as a dictionary
    return {'die': die, 'survive': survive}

def feature_weights(self):
    # extract the feature importances from the decision tree model
    importances = self.dt.feature_importances_
    # return the feature importances as a dictionary, using dictionary comprehension
    return {feature: importance for feature, importance in zip(self.features, importances)}

def initTitanic():
    TitanicModel.get_instance()
    
def testTitanic():
    """ Test the Titanic Model
    Using the TitanicModel class, we can predict the survival probability of a passenger.
    Print output of this test contains method documentation, passenger data, survival probability, and survival weights.
    """
     
    # setup passenger data for prediction
    print(" Step 1:  Define theoretical passenger data for prediction: ")
    passenger = {
        'name': ['Sumedha Kamaraju'],
        'pclass': [2],
        'sex': ['female'],
        'age': [17],
        'sibsp': [0],
        'parch': [2],
        'fare': [16.00],
        'embarked': ['S'],
        'alone': [False]
    }
    print("\t", passenger)
    print()
    
    # get an instance of the cleaned and trained Titanic Model
    titanicModel = TitanicModel.get_instance()
    print(" Step 2:", titanicModel.get_instance.__doc__)
   
    # print the survival probability
    print(" Step 3:", titanicModel.predict.__doc__)
    probability = titanicModel.predict(passenger)
    print('\t death probability: {:.2%}'.format(probability.get('die')))  
    print('\t survival probability: {:.2%}'.format(probability.get('survive')))
    print()
    
    # print the feature weights in the prediction model
    print(" Step 4:", titanicModel.feature_weights.__doc__)
    importances = titanicModel.feature_weights()
    for feature, importance in importances.items():
        print("\t\t", feature, f"{importance:.2%}") # importance of each feature, each key/value pair
        
if __name__ == "__main__":
    print(" Begin:", testTitanic.__doc__)
    testTitanic()
    
    titanic_api = Blueprint('titanic_api', __name__,
                   url_prefix='/api/titanic')

    api = Api(titanic_api)
    
    class TitanicAPI:
        class _Predict(Resource):
            def post(self):
                # Get the passenger data from the request
                passenger = request.get_json()

                # Get the singleton instance of the TitanicModel
                titanicModel = TitanicModel.get_instance()
                # Predict the survival probability of the passenger
                response = titanicModel.predict(passenger)

                # Return the response as JSON
                return jsonify(response)

        api.add_resource(_Predict, '/predict')