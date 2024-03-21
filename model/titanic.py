import numpy as np

## Python Titanic API endpoint
from flask import request, jsonify

# Define the API endpoint for prediction
@app.route('/api/predict', methods=['POST'])
def predict():
    # Get the passenger data from the request
    passenger = request.get_json()

    response = predictSurvival(passenger)

    # Return the response as JSON
    return jsonify(response)

def initTitanicData():
    # Logistic regression model is used to predict the probability

# Define a new passenger
    passenger = pd.DataFrame({
    'name': ['Sumedha Kamaraju'],
    'pclass': [2], # 2nd class picked as it was median, bargains are my preference, but I don't want to have poor accomodations
    'sex': ['female'],
    'age': [17],
    'sibsp': [0], # I usually travel with my wife
    'parch': [2], # currenly I have 1 child at home
    'fare': [16.00], # median fare picked assuming it is 2nd class
    'embarked': ['S'], # majority of passengers embarked in Southampton
    'alone': [False] # travelling with family (spouse and child))

    display(passenger)
new_passenger = passenger.copy()

# Preprocess the new passenger data
new_passenger['sex'] = new_passenger['sex'].apply(lambda x: 1 if x == 'male' else 0)
new_passenger['alone'] = new_passenger['alone'].apply(lambda x: 1 if x == True else 0)

# Encode 'embarked' variable
onehot = enc.transform(new_passenger[['embarked']]).toarray()
cols = ['embarked_' + val for val in enc.categories_[0]]
new_passenger[cols] = pd.DataFrame(onehot, index=new_passenger.index)
new_passenger.drop(['name'], axis=1, inplace=True)
new_passenger.drop(['embarked'], axis=1, inplace=True)

display(new_passenger)

# Predict the survival probability for the new passenger
dead_proba, alive_proba = np.squeeze(logreg.predict_proba(new_passenger))

# Print the survival probability
print('Death probability: {:.2%}'.format(dead_proba))  
print('Survival probability: {:.2%}'.format(alive_proba))

})

