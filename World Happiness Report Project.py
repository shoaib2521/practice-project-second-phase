#!/usr/bin/env python
# coding: utf-8

# #                               World Happiness Report Project

# In[15]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

url = "https://github.com/FlipRoboTechnologies/ML-Datasets/raw/main/World%20Happiness/happiness_score_dataset.csv"
df = pd.read_csv(url)

print(df.head())

print(df.isnull().sum())

X = df[['GDP per Capita', 'Family', 'Life Expectancy', 'Freedom', 'Generosity', 'Trust Government Corruption']]
y = df['Happiness Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"Root Mean Squared Error: {rmse}")

new_data = [[1.3, 1.4, 0.7, 0.2, 0.1, 0.1]]
predicted_score = model.predict(new_data)
print(f"Predicted Happiness Score: {predicted_score[0]}")


# In[18]:


print(df.columns)

X = df[['Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)', 'Freedom', 'Generosity', 'Trust (Government Corruption)']]
y = df['Happiness Score']

print(df.head())


# In[ ]:




