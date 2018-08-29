import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from data_missing import set_missing_ages, set_Cabin_type
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from feature_engineering_4_train import llr

data_test = pd.read_csv('J:/Code/kaggle/Titanic/test.csv')
data_test.info()
pd.set_option('display.max_rows',None)
print(data_test['PassengerId'])
##发现test数据集的fare有缺失
# 找到缺失的坐标
# print(data_test.loc[data_test['Fare'].isnull().values == True])
data_test.loc[data_test['Fare'].isnull().values == True, 'Fare'] = 0

# print(data_test['PassengerId'])
data_test, _ = set_missing_ages(data_test)

# print(data_test)

data_test = set_Cabin_type(data_test)

# print(data_test)

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

# print('test_df : \n', test_df)

prediction = llr.predict(test_df)

print(data_test['PassengerId'])
result = pd.DataFrame({'PassengerId' : data_test['PassengerId'].as_matrix(), 'Survived' : prediction.astype(np.int32)})

result.to_csv("J:/Code/kaggle/Titanic/logistic_regression_predictions.csv", index = False)