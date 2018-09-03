# 0.76555 not better than before
import pandas as pd
import os
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from data_missing import set_missing_ages_mean, set_missing_ages_rf, set_Cabin_type
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
# from sklearn import cross_validation
# from sklearn.utils import shuffle
# from plt_learning_curve import plot_learning_curve
from sklearn.ensemble import BaggingRegressor

print('training code is processing : ')
# os.system("pause")

data_train = pd.read_csv('J:/Code/kaggle/Titanic/train.csv')
# data_train.info()

#### data analysis


##################################################


######################## featrue engineering
if __name__ == '__main__':

    data_train = set_missing_ages_mean(data_train)

    # print(rfr)

    data_train = set_Cabin_type(data_train)

    # print(data_train)

    # 将需要的类目属性全部转换成0,1的数值

    dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix = 'Cabin')

    dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix = 'Embarked')

    dummies_Sex = pd.get_dummies(data_train['Sex'], prefix = 'Sex')

    dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix = 'Pclass')

    df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis = 1)
    # print('df : \n', df)

    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis = 1, inplace = True)
    # print('df : \n', df)
    # print(pd.concat([df['Age'], df['Fare']], axis = 1))

    scaler = preprocessing.StandardScaler()

    # 标准化，调整为均值为0方差为1的标准正太分布
    age_scale_param = scaler.fit(df[['Age']])
    df['Age_scaled'] = scaler.fit_transform(df[['Age']], age_scale_param)
    # print('df[\'Age_scaled\'] : \n', df['Age_scaled'])

    fare_scale_param = scaler.fit(df[['Fare']])
    df['Fare_scaled'] = scaler.fit_transform(df[['Fare']], age_scale_param)
    # print('df[\'Fare_scaled\'] : \n', df['Fare_scaled'])
    # print(pd.concat([df['Age_scaled'], df['Fare_scaled']], axis = 1))

    ### building the logistic-regression model

    # 正则表达式取出想要的属性值
    train_df = df.filter(regex = 'PassengerId|Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.\D|Embarked_[^0]|Sex_.\D|Pclass_[1-3]')

    train_df = train_df.as_matrix()
    # label : survived
    y = train_df[:, 1]

    # featrue : Parch, Cabin_No, Cabin_Yes...
    X = train_df[:, 2:]

    # baggin g model
    # print('train the bagging model : ')
    # os.system("pause")

    llr = linear_model.LogisticRegression(C = 1.0, penalty = 'l1', tol = 1e-6)

    bagging_llr = BaggingRegressor(llr, n_estimators = 20, max_samples = 0.8, max_features = 1.0, bootstrap = True, bootstrap_features = False, n_jobs = -1)
    
    bagging_llr.fit(X, y)
    
    print('training is over')
    # os.system("pause")
    
    ###### test data
    
    data_test = pd.read_csv('J:/Code/kaggle/Titanic/test.csv')
    data_test.info()

    print(data_test['PassengerId'])
    ##发现test数据集的fare有缺失
    # 找到缺失的坐标
    # print(data_test.loc[data_test['Fare'].isnull().values == True])
    data_test.loc[data_test['Fare'].isnull().values == True, 'Fare'] = 0

    # print(data_test['PassengerId'])
    data_test = set_missing_ages_mean(data_test)

    # print(data_test)

    data_test = set_Cabin_type(data_test)

    # unknown_age = data_test[data_test.Age.isnull()]

    # print(unknown_age)

    # os.system("pause")

    ###将Cabin, Embarked, Sex, Pclass进行特征表征0,1化处理

    dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix = 'Cabin')

    dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix = 'Embarked')

    dummies_Sex = pd.get_dummies(data_test['Sex'], prefix = 'Sex')

    dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix = 'Pclass')

    df = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis = 1)

    # print(df.filter(regex = 'Sex|Sex_.*'))

    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis = 1, inplace = True)

    # print(df)

    ### 对Age和Fare进行标准化处理
    scaler = preprocessing.StandardScaler()

    age_scale_param = scaler.fit(df[['Age']])
    df['Age_scaled'] = scaler.fit_transform(df[['Age']], age_scale_param)
    # print('df[\'Age_scaled\'] : \n', df['Age_scaled'])

    fare_scale_param = scaler.fit(df[['Fare']])
    df['Fare_scaled'] = scaler.fit_transform(df[['Fare']], age_scale_param)
    # print('df[\'Fare_scaled\'] : \n', df['Fare_scaled'])

    #### 取出想要的属性
    test_df = df.filter(regex = 'Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.\D|Embarked_[^0]|Sex_.\D|Pclass_[1-3]')

    pd.set_option('display.max_columns',None)

    print('test_df : \n', test_df)
    # print(llr)
    os.system("pause")

    # not using bagging
    # prediction = llr.predict(test_df)

    # bagging
    prediction = bagging_llr.predict(test_df)

    print(data_test['PassengerId'])
    result = pd.DataFrame({'PassengerId' : data_test['PassengerId'].as_matrix(), 'Survived' : prediction.astype(np.int32)})

    result.to_csv("J:/Code/kaggle/Titanic/logistic_regression_predictions_bag.csv", index = False)
