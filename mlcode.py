  
import pandas as pd
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('50_startups.csv')

#print(dataset)
y = dataset['Profit']
y = y.values

X =  dataset[ ['R&D Spend', 'Administration', 'Marketing Spend', 'State' ] ]

X.shape

#print(X)
#Label Encoding
state  = dataset['State']
state_le = LabelEncoder()
state = state_le.fit_transform(state)
state = state.reshape(-1, 1)
#print(state.shape)


#One Hot Encoding
state_ohe = OneHotEncoder()
state_dummy = state_ohe.fit_transform(state)
state_final = state_dummy.toarray()
state_final = state_final[: , 0:2 ]

X = X.values
X = X[: , 0:3 ]

X_final =  numpy.hstack(  (X,  state_final))


#fiting model and predicting our output

model = LinearRegression()
#Prediction
x_train, x_test , y_train , y_test =train_test_split(X_final, y, test_size=0.3)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("This is our 1st predction given by model : \n", y_pred[0])
