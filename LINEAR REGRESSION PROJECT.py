#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[26]:


df=pd.read_csv("salary_dataset.csv")
df


# In[98]:


df.head()


# In[ ]:


#scatter plot
plt.scatter(df['YearsExperience'],df['Salary'])
plt.xlabel('YearsExperience')
plt.ylabel('Salary')


# In[96]:


##correlation
df.corr()


# In[42]:


##seaborn for visualization
import seaborn as sns
sns.pairplot(df)


# In[43]:


##independent and dependent features
X=df[['YearsExperience']]##independent feature
y=df[['Salary']]##dependent feature


# In[44]:


X_series=df['YearsExperience']

np.array(x_series).shape


# In[45]:


np.array(y).shape


# In[46]:


##Train Test Split
from sklearn.model_selection import train_test_split


# In[47]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


# In[48]:


X_train.shape


# In[52]:


##standardization
from sklearn.preprocessing import StandardScaler


# In[53]:


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)


# In[54]:


X_test=scaler.transform(X_test)


# In[55]:


X_test


# In[57]:


##Apply simple linear regression
from sklearn.linear_model import LinearRegression


# In[63]:


regression=LinearRegression(n_jobs=-1)


# In[64]:


regression.fit(X_train,y_train)


# In[68]:


print('coefficient of slope:',regression.coef_)
print('intercept:',regression.intercept_)


# In[71]:


#plot Training data plot for best fit line
plt.scatter(X_train,y_train)
plt.plot(X_train,regression.predict(X_train))


# In[82]:


##prediction for the test data
y_pred=regression.predict(X_test)


# In[83]:


##performance metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error 


# In[84]:


mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mse)
print(mse)
print(mae)
print(rmse)


# In[85]:


from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print(score)


# In[89]:


#display adjusted R-Squared
1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)



# In[91]:


#OLS Linear Regression
import statsmodels.api as sm


# In[92]:


model=sm.OLS(y_train,X_train).fit()
prediction=model.predict(X_test)
print(prediction)


# In[93]:


print(model.summary())


# In[100]:


##prediction for new data
regression.predict(scaler.transform([[20]]))


# In[ ]:




