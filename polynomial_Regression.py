import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
#IMPORTING LIBRARIES

df=pd.read_csv("Position_Salaries.csv")
#IMPORTING CSV FILE


#%%

df.head()
#FIRST 5 ROWS

x=df["Level"].values.reshape(-1,1)
y=df["Salary"].values.reshape(-1,1)
#X AND Y AXIS


#%%
plt.scatter(x,y)
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.xticks(np.arange(1,10,0.5)) 
plt.show() 
#VISUALISING DATAS


#%%
lr=LinearRegression()
lr.fit(x, y)
#FITTING LINEAR REGRESSION TO THE DATASET


pl=PolynomialFeatures(degree=4)
lr2=LinearRegression()
x_pl=pl.fit_transform(x)
lr2.fit(x_pl,y)
#FITTING POLYNOMIAL  REGRESSION TO THE DATASET


lr_predict=lr.predict(x)#lr values
pl_predict=lr2.predict(x_pl)#pl values
#PREDICTING Y VALUES WITH LINEAR AND POLYNOMIAL REGRESSION


#%%
plt.scatter(x, y, color = 'blue')
plt.plot(x, lr_predict, color = 'red')
plt.title('Linear Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
#VISUALISING LINEAR REGRESSION RESULTS

#%%
plt.scatter(x, y, color = 'blue')
plt.plot(x, pl_predict, color = 'red')
plt.title('Polynomial  Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
#VISUALISING POLYNOMIAL REGRESSION RESULTS


#%%
#Predicting a new result with Linear Regression
print(lr.predict(6.5))


#%%
#Predicting a new result with Polynomial Regression
print(lr2.predict(pl.fit_transform([[6.5]])))



























