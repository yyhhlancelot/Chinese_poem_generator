# logistic regression model for Titanic survivor prediction

import pandas as pd
import os
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from data_missing import set_missing_ages_mean, set_missing_ages_rf, set_Cabin_type
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.utils import shuffle
from plt_learning_curve import plot_learning_curve

print('training code is processing : ')
# os.system("pause")

data_train = pd.read_csv('J:/Code/kaggle/Titanic/train.csv')
data_train.info()

#### data analysis

fig = plt.figure()
fig.set(alpha = 0.2)

############################ 画图分析
plt.subplot2grid((2, 3), (0, 0))
data_train.Survived.value_counts().plot(kind = 'bar')
plt.title('survivor_distribution')
plt.ylabel('population')

plt.subplot2grid((2, 3), (0, 1))
data_train.Pclass.value_counts().plot(kind = 'bar')
plt.title('people_class')
plt.ylabel('distribution form Pclass')

plt.subplot2grid((2, 3), (0, 2))
plt.scatter(data_train.Survived, data_train.Age)
# data_train.Survived.value_counts().plot(kind = 'bar')
plt.grid(b = True, which = 'major', axis = 'y')
plt.title('distribution from age')
plt.ylabel('age')

plt.subplot2grid((2, 3), (1, 0), colspan = 2)
data_train.Age[data_train.Pclass == 1].plot(kind = 'kde')
data_train.Age[data_train.Pclass == 2].plot(kind = 'kde')
data_train.Age[data_train.Pclass == 3].plot(kind = 'kde')
plt.xlabel('age')
plt.ylabel('density')
plt.title('class-age distribution')
plt.legend(('first', 'second', 'third'), loc = 'best')

plt.subplot2grid((2, 3), (1, 2))
data_train.Embarked.value_counts().plot(kind = 'bar')
plt.title('terminal')
plt.ylabel('population')
# plt.show()

# fig2 = plt.figure()
# fig2.set(alpha = 0.2)

# fig = plt.figure()
# fig.set(alpha = 0.2)
# ax1=fig.add_subplot(121)
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({'survived' : Survived_1, 'not survived' : Survived_0})
df.plot(kind = 'bar')
plt.title('class-surviving')
plt.xlabel('passenger-class')
plt.ylabel('population')
# plt.show()

# ax2=fig.add_subplot(122)
Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df = pd.DataFrame({'male' : Survived_m, 'female' : Survived_f})
df.plot(kind = 'bar', stacked = True)
plt.title('sex-surviving')
plt.xlabel('sex')
plt.ylabel('population')
# plt.show()

fig = plt.figure()
plt.title("class-sex surviving")

ax1=fig.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female high class", color='#FA2479')
ax1.set_xticklabels(["not survived", "survived"], rotation=0)
ax1.legend(["female/highclass"], loc='best')

ax2=fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
ax2.set_xticklabels(["not survived", "survived"], rotation=0)
plt.legend(["female/low class"], loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
ax3.set_xticklabels(["not survived", "survived"], rotation=0)
plt.legend(["male/highclass"], loc='best')

ax4=fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
ax4.set_xticklabels(["not survived", "survived"], rotation=0)
plt.legend(["male/low class"], loc='best')
# plt.show()

Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df=pd.DataFrame({'survived':Survived_1, 'not survived':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title("terminal-surviving")
plt.xlabel("terminal") 
plt.ylabel("population") 

# plt.show()
##################################################


######################## featrue engineering

data_train = set_missing_ages_mean(data_train)

# print(rfr)

data_train = set_Cabin_type(data_train)

print(data_train)

# 将需要的类目属性全部转换成0,1的数值

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix = 'Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix = 'Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix = 'Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix = 'Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis = 1)
print('df : \n', df)

df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis = 1, inplace = True)
print('df : \n', df)
print(pd.concat([df['Age'], df['Fare']], axis = 1))

scaler = preprocessing.StandardScaler()

# 标准化，调整为均值为0方差为1的标准正太分布
age_scale_param = scaler.fit(df[['Age']])
df['Age_scaled'] = scaler.fit_transform(df[['Age']], age_scale_param)
print('df[\'Age_scaled\'] : \n', df['Age_scaled'])

fare_scale_param = scaler.fit(df[['Fare']])
df['Fare_scaled'] = scaler.fit_transform(df[['Fare']], age_scale_param)
print('df[\'Fare_scaled\'] : \n', df['Fare_scaled'])
print(pd.concat([df['Age_scaled'], df['Fare_scaled']], axis = 1))

### building the logistic-regression model

# 正则表达式取出想要的属性值
train_df = df.filter(regex = 'PassengerId|Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.\D|Embarked_[^0]|Sex_.\D|Pclass_[1-3]')

### training_set : validation_set = 7 : 3
split_train_df, split_cv_df = cross_validation.train_test_split(train_df, test_size = 0.3, random_state = 0)

# pd.set_option('display.max_columns',None)
print(train_df)
# os.system("pause")

train_np = train_df.as_matrix()
print(train_np)

split_train_df = split_train_df.as_matrix()

# label : survived
y = split_train_df[:, 1]

# featrue : Parch, Cabin_No, Cabin_Yes...
X = split_train_df[:, 2:]

llr = linear_model.LogisticRegression(C = 1.0, penalty = 'l1', tol = 1e-6)

llr.fit(X, y)

print(llr)

print(pd.DataFrame({"columns" : list(train_df.columns)[2:], "coef" : list(llr.coef_.T)}))

os.system("pause")
########## validation set/cross validation

### training_set : validation_set = 7 : 3

# llr.fit(split_train_df.as_matrix()[:, 2:], split_train_df.as_matrix()[:, 1]) # 这里是因为加入了passengerid所以从2开始

### predict validation set
# split_cv_df = split_cv_df.filter(regex = 'Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.\D|Embarked_[^0]|Sex_.\D|Pclass_[1-3]')
prediction_cv = llr.predict(split_cv_df.as_matrix()[:, 2:])

### 预测失败的validation

print(prediction_cv)
print(split_cv_df)
print(split_cv_df[prediction_cv != split_cv_df.values[:, 1]])#['PassengerId'].values)


ori_data_train = pd.read_csv('J:/Code/kaggle/Titanic/train.csv')

bad_cases = ori_data_train.loc[ori_data_train['PassengerId'].isin(split_cv_df[prediction_cv != split_cv_df.values[:, 1]]['PassengerId'].values)]

bad_cases = bad_cases.filter(regex = 'PassengerId|Survived|Name|Age|SibSp|Parch|Fare|Cabin|Embarked|Sex|Pclass')

pd.set_option('display.max_columns',None)

print('bad_cases : \n', bad_cases)
print(bad_cases.iloc[:, 0].size)

########### analysis of bad_cases
bad_cases_male = bad_cases.Survived[bad_cases.Sex == 'male'].value_counts()
print(bad_cases_male)
bad_cases_female = bad_cases.Survived[bad_cases.Sex == 'female'].value_counts()
print(bad_cases_female)
# survived_0 = bad_cases.Survived[bad_cases.Survived == 0].value_counts()
# survived_1 = bad_cases.Survived[bad_cases.Survived == 1].value_counts()
# df = pd.DataFrame({'survived' : survived_1, 'not survived' : survived_0})

## 该图发现bad_case里面很多男的活了下来，女的死了，没有遵循模型内部的女士优先原则

df = pd.DataFrame({'bad_cases_female' : bad_cases_female, 'bad_cases_male' : bad_cases_male})
df.plot(kind = 'bar')
plt.title('bad_cases-surviving')
# plt.xlabel('survived or not')
plt.xlabel('sex')
plt.ylabel('population')
# plt.show()
print("X_size :",X[:, 0].size)
print("y_size :",y[:].size)
X_shuf, y_shuf = shuffle(X, y)
plot_learning_curve(llr, "learning curve", X_shuf, y_shuf)
# plot_learning_curve(llr, "learning curve", X, y)