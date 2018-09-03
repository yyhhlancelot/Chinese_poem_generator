from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import os

def set_missing_ages_mean(df):
    '''use randomforest to fill the missing data'''
    # print('df : \n', df)
    age_df = df[['Name', 'Age', 'Fare', 'Parch', 'SibSp', 'Pclass']] #将上述项从dataframe里面提取出来
    
    ###############################################
    ############ 通过Name中的信息来对未知年龄使用平均值代替 ##############
    known_age = age_df[age_df.Age.notnull()]
    
    unknown_age = age_df[age_df.Age.isnull()]
    ### 平均Mr的年龄
    known_age_Mr = known_age[known_age['Name'].str.contains('Mr\.')]
    
    unknown_age_Mr = unknown_age[unknown_age['Name'].str.contains('Mr\.')]
    
    unknown_age_Mr['Age'].fillna(known_age_Mr['Age'].mean(), inplace = True)
    
    age_df.iloc[unknown_age_Mr.index.tolist(), :] = unknown_age_Mr
    
    ### 平均Mrs的年龄
    known_age_Mrs = known_age[known_age['Name'].str.contains('Mrs\.')]
    
    unknown_age_Mrs = unknown_age[unknown_age['Name'].str.contains('Mrs\.')]
    
    unknown_age_Mrs['Age'].fillna(known_age_Mrs['Age'].mean(), inplace = True)
    
    age_df.iloc[unknown_age_Mrs.index.tolist(), :] = unknown_age_Mrs
    
    ### 平均Miss的年龄
    known_age_Miss = known_age[known_age['Name'].str.contains('Miss\.')]
    
    unknown_age_Miss = unknown_age[unknown_age['Name'].str.contains('Miss\.')]
    
    unknown_age_Miss['Age'].fillna(known_age_Miss['Age'].mean(), inplace = True)
    
    age_df.iloc[unknown_age_Miss.index.tolist(), :] = unknown_age_Miss
    
    ### 平均Dr的年龄
    known_age_Dr = known_age[known_age['Name'].str.contains('Dr\.')]
    
    unknown_age_Dr = unknown_age[unknown_age['Name'].str.contains('Dr\.')]
    
    unknown_age_Dr['Age'].fillna(known_age_Dr['Age'].mean(), inplace = True)
    
    age_df.iloc[unknown_age_Dr.index.tolist(), :] = unknown_age_Dr
    
    ### 平均Master的年龄
    known_age_Master = known_age[known_age['Name'].str.contains('Master\.')]
    
    unknown_age_Master = unknown_age[unknown_age['Name'].str.contains('Master\.')]
    
    unknown_age_Master['Age'].fillna(known_age_Master['Age'].mean(), inplace = True)
    
    age_df.iloc[unknown_age_Master.index.tolist(), :] = unknown_age_Master
    ### test里面出现了Ms.年龄缺失
    unknown_age_Ms = unknown_age[unknown_age['Name'].str.contains('Ms\.')]
    
    unknown_age_Ms['Age'].fillna(40, inplace = True)
    
    age_df.iloc[unknown_age_Ms.index.tolist(), :] = unknown_age_Ms
    
    print('df before: \n', df)
    
    print('age_df : \n', age_df)
    
    # df.iloc[age_df.index.tolist(), ['Name', 'Age', 'Fare', 'Parch', 'SibSp', 'Pclass']] = age_df
    
    
    
    df.loc[age_df.index.tolist(), ['Name', 'Age', 'Fare', 'Parch', 'SibSp', 'Pclass']] = age_df
    
    print('df after: \n', df)
    
    # os.system("pause")
    
    return df
    
def set_missing_ages_rf(df):
    ##############################################
    ########### 使用随机森林来拟合年龄 ###########
    age_df = df[['Name', 'Age', 'Fare', 'Parch', 'SibSp', 'Pclass']] #将上述项从dataframe里面提取出来
    
    known_age = age_df[age_df.Age.notnull()]
    
    unknown_age = age_df[age_df.Age.isnull()]
    
    y = known_age.values[:, 0]
    
    # print('y : ', y)
    
    X = known_age.values[:, 1:]
    
    # print('X : ', X)
    
    rfr = RandomForestRegressor(random_state = 0, n_estimators = 2000, n_jobs = -1)
    
    rfr.fit(X, y) # 随机森林回归
    
    print('unknown_age.values : \n', unknown_age.values)
    
    predictedAges = rfr.predict(unknown_age.values[:, 1:])
    
    print('predictedAges : \n', predictedAges)
    
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges
    
    return df
    ##############################################
    
def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df
    
