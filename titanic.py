# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 14:08:07 2017

@author: Administrator
"""

# I don't have so much knowledge about titanic
# Base on my instinct to do analysis

pwd

# Basic description of data (Exploratory Analysis)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# read the data
df = pd.read_csv('titanic.csv')

df.shape # Dimension
df.head() # first 5 rows
df.tail() # last 5 rows

header = list(df.columns.values) # get the list of column names

df.info() # non-null count, metadata, we can see there is null data in "Age", "Cabin", "Embarked"
df.describe() # Summary Statistics, we can see the survival rate
df.skew # Highly right-skew for fare, need to take care

df.isnull().any() # See what column has null value
df.isnull().sum() # See how many null value in each column

df.hist() # All histogram which can shown

# Plot density graph to check the distribution of continuous data
df['Age'].plot(kind = 'density')
df['Fare'].plot(kind = 'density')

# Boxplot for continuous data
sns.boxplot(x = df['Age'])
sns.boxplot(x = df['Fare'])

sns.boxplot(x = "Survived", y = "Fare", data = df)
sns.boxplot(x = "Survived", y = "Age", data = df)

sns.boxplot(x = "Pclass", y = "Fare", data = df)
sns.boxplot(x = "Pclass", y = "Age", data = df)

sns.boxplot(x = "Embarked", y = "Fare", data = df)
sns.boxplot(x = "Embarked", y = "Age", data = df)

sns.boxplot(x = "Sex", y = "Fare", data = df)
sns.boxplot(x = "Sex", y = "Age", data = df)


# Make Correlation graph
corr = df.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
# We can find some insights here:
# Higher class has higher survival rate
# Higher fare has higher survival rate (may same as above)
# Higher age, higher class (make sense)
# Higher fare, higher class (make sense)

# First, need to confirm higher class is higher fare
sns.barplot(x = 'Pclass', y = 'Fare', data = df)
# 1st class fare: 4 times of 2nd class fare; 6 times of 3rd class fare
# So, we can confirm that: higher class, higher survial rate 

# Second, a strange thought, for top/bottom rich people, are they survived?
df.nlargest(10, 'Fare')[['Pclass', 'Fare', 'Survived']] # Top 10 rich, all in class 1, high survival rate
df.nsmallest(10, 'Fare')[['Pclass', 'Fare', 'Survived']] # Bottom 10 rich, some class 1 here (dunno why), low survival rate

# we should exclude class 1 for richest & class 3 for poorest
# Let's define a function for top x people for richest/poorest ranking

from fareRanking import top_rich
top_rich(df, 50, 'rich', True) 
top_rich(df, 50, 'poor', True)

#==============================================================================
# # Debug for reloading the module:
# # import imp
# # imp.reload(fareRanking) 
#==============================================================================

# Count Plot for categorical data
sns.countplot(x = 'Sex', data = df) 
sns.countplot(x = 'Survived', data = df)
print("Survial Rate is: %f" % df['Survived'].mean()) # Survival Rate
sns.countplot(x = 'SibSp', data = df)
sns.countplot(x = 'Pclass', data = df)
sns.countplot(x = 'Parch', data = df)
sns.countplot(x = 'Embarked', data = df)

# Let's have some basic analysis
# In titanic, they let female and young leave first, is that true?
sns.barplot(x = 'Sex', y = 'Survived' , data = df) # How much proportion for male/female survived
# From above graph, near 20% males survived, while more than 70% female survived, seems solid
# But how many of the male is young? we need to define what is young first
# We create a new feature: Adult

# I set age > 16 as True, otherwise False
df['Adult'] = df['Age'] > 16
sns.countplot(x = 'Adult', data = df) # ~30% is young

# Let's see the survival rate for adult and young
sns.barplot(x = 'Adult', y = 'Survived' , data = df)
# It looks similar, the chance for adult and young survival are nearly the same (~38%)

# How about young male? Can they have higher chance for survival?
sns.barplot(x = 'Adult', y = 'Survived' , data = df[df['Sex'] == 'male'])
# Amoung male, adult only has 17% survival rate, while young has around 22%
# That's means, if you are a male adult, probably you will die

# Also, for different classes, is that richer class has higher survival rate?
sns.barplot(x = 'Pclass', y = 'Survived' , data = df)
# Clearly, more than 60% of 1st class people survive. Sad for poor. 

# Third, How about the kids/elderly? Are they survived?
# We create 2 more features: kids, elderly
# For kids, Age <= 10 is True, otherwise is False
# For elderly, Age >=50 is True, otherwise is False (life-span shorter at that time)

df['kids'] = df['Age'] <= 10
df['elderly'] = df['Age'] >= 50
sns.barplot(x = 'kids', y = 'Survived' , data = df) # Around 60%, not bad
sns.barplot(x = 'elderly', y = 'Survived' , data = df) # Around 36%, similar to normal person

# We may make a small conclusion here:
# kids has higher chance for survival, while other age group is similar chance for survival
# rich has higher change for survival than poor
# Female is more lucky than male ( 70% vs 20%)
# Male adult, pray yourself (17%)

#############################################################################################
# Let's PLay some Machine Learning!

# We drop some features (at this moment, no feature engineering) that seems useless
df_2 =  df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)

# 1st: Features Scaling
# Create X,y
X = df_2.drop('Survived', axis = 1)
y = df_2['Survived']

# Log transformation for Fare
X['Fare'] = np.log10(df_2['Fare']+5)

# For Age, fill mean value of age to 'nan'
X['Age'] = X['Age'].fillna(X['Age'].mean())

# For categorical data, transform to numerical
X.dtypes # Check dtype
from toCategory import toCategory
# transform to categrical dtype, and then assign to numerical
toCategory(X, ['Sex', 'Embarked', 'Pclass']) 
toCategory(X, ['Adult', 'kids', 'elderly']) 



#==============================================================================
# Feature Scaleing
# from sklearn.preprocessing import MinMaxScaler, scale
# scaler = MinMaxScaler(feature_range = (0,1))
# rescaledX = scaler.fit_transform(array)


# scale(array)
#==============================================================================



from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression(C = 0.07)
logistic.fit(X,y)

logistic.predict(X)
logistic.score(X,y)


from sklearn.model_selection import learning_curve
train_sizes, train_scores, valid_scores = learning_curve(LogisticRegression(), X, y, cv=5)


from learningCurve import plot_learning_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression


title = "Learning Curves (Logistic Regression)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = LogisticRegression(C = 0.07)
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10))
plt.show()


X = X.drop(['kids','Adult','elderly'], axis = 1)







##################################################################################
# Test Set
df_test = pd.read_csv('test_titanic.csv')

df_test2 =  df_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)

# 1st: Features Scaling
# Create X,y
X_test = df_test2

# Log transformation for Fare
X_test['Fare'] = np.log10(df_test2['Fare']+5)

# For Age, fill mean value of age to 'nan'
X_test['Age'] = X_test['Age'].fillna(X_test['Age'].mean())
X_test['Fare'] = X_test['Fare'].fillna(X_test['Fare'].mean())

# For categorical data, transform to numerical
X.dtypes # Check dtype
from toCategory import toCategory
# transform to categrical dtype, and then assign to numerical
toCategory(X_test, ['Sex', 'Embarked', 'Pclass']) 


logistic = LogisticRegression(C = 0.07)
logistic.fit(X,y)

y_test = logistic.predict(X_test)


result = pd.DataFrame(y_test, columns = ['Survived'])
result['PassengerId'] = df_test['PassengerId']
result = result[['PassengerId', 'Survived']]
result.to_csv('y_test_titanic3.csv',index = False)






















