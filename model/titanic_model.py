def initTitanicData():
    print("Titanic Data")


    print(titanic_data.columns) # titanic data set
display(titanic_data[['survived','pclass', 'sex', 'age', 'sibsp', 'parch', 'class', 'fare', 'embark_town', 'alone']]) # look at selected columns  # noqa: F821
print(td.columns)
display(td)  # noqa: F821

print(titanic_data.median())
print(titanic_data.query("survived == 0").mean())
print(td.query("survived == 1").mean())
print("maximums for survivors")
print(td.query("survived == 1").max())
print()
print("minimums for survivors")
print(td.query("survived == 1").min())
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('DecisionTreeClassifier Accuracy: {:.2%}'.format(accuracy))  
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('LogisticRegression Accuracy: {:.2%}'.format(accuracy))  