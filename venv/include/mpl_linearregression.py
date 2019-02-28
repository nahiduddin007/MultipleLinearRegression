
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values



#Encoding Catagorial data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy variable Trap(n-1)
X = X[:,1:]


# Spliting the dataset into the tranning set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2,random_state=0)


#Fitting Multiple linear Regression to Tranning set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor=regressor.fit(X_train, y_train)


#Building the optimal model using backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressorOLS = sm.OLS(endog = y, exog = X_opt).fit()
regressorOLS.summary()


X_opt = X[:, [0, 1, 3, 4, 5]]
regressorOLS = sm.OLS(endog = y, exog = X_opt).fit()
regressorOLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regressorOLS = sm.OLS(endog = y, exog = X_opt).fit()
regressorOLS.summary()

X_opt = X[:, [0, 3, 5]]
regressorOLS = sm.OLS(endog = y, exog = X_opt).fit()
regressorOLS.summary()

X_opt = X[:, [0, 3]]
regressorOLS = sm.OLS(endog = y, exog = X_opt).fit()
regressorOLS.summary()



print(X)
print(X_opt)
print(X_test)
print(X_train)
print(dataset)
print(y)
print(y_test)
print(y_train)
