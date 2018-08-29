from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

def set_missing_ages(df):
    '''use randomforest to fill the missing data'''
    # print('df : \n', df)
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']] #将上述项从dataframe里面提取出来
    
    # known_age = age_df[age_df.Age.notnull()].values()
    known_age = age_df[age_df.Age.notnull()]
    # print('known_age : \n', known_age)
    # unknown_age = age_df[age_df.Age.isnull()].values()
    unknown_age = age_df[age_df.Age.isnull()]
    # print('unknown_age : \n', unknown_age)
    y = known_age.values[:, 0]
    # print('y : ', y)
    X = known_age.values[:, 1:]
    # print('X : ', X)
    rfr = RandomForestRegressor(random_state = 0, n_estimators = 2000, n_jobs = -1)
    
    rfr.fit(X, y) # 随机森林回归
    
    # print('unknown_age.values : \n', unknown_age.values)
    
    predictedAges = rfr.predict(unknown_age.values[:, 1:])
    
    # print('predictedAges : \n', predictedAges)
    
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges
    
    return df, rfr
    
def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df
    
