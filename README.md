# logistic regression for Titanic event

## Brief introduction
train a logistic regression model to predict the surviving situation in Titanic event.

logistic_regression_4_train.py includes data preprocessing and prefeature engineering, and train a logistic regression model to fit the data. 

data_missing.py includes function to deal the missing data(NaN).

logistic_regression_4_test.py uses the model trained before to predict the label of the test data.

plt_learning_curve.py includes a function to draw the learing curve of train set and validation set, to determine whether the model is overfitting or underfitting.

bagging_regressor.py build a bagging regressor to deal the overfitting situation.