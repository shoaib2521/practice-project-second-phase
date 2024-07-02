                                  #Titanic survived Project


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
​
url = "https://github.com/FlipRoboTechnologies/ML-Datasets/raw/main/Titanic/titanic_train.csv"
df = pd.read_csv(url)
​
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
​
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df)
plt.title('Distribution of Survived')
plt.show()
​
plt.figure(figsize=(6, 4))
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Sex')
plt.show()
​
plt.figure(figsize=(6, 4))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Pclass')
plt.show()
​
plt.figure(figsize=(6, 4))
sns.barplot(x='Embarked', y='Survived', data=df)
plt.title('Survival Rate by Embarked')
plt.show()
​
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
​
X = df.drop('Survived', axis=1)
y = df['Survived']
​
from sklearn.model_selection import train_test_split
​
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
​
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
​
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
​
rf_model = RandomForestClassifier(random_state=42)        
rf_model.fit(X_train, y_train)
​
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
​
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
​
