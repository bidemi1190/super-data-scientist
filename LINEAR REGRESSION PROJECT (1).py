#!/usr/bin/env python
# coding: utf-8

# #The objective here was straightforward: to see the effect years experience has on salary theand there was a strong positive correlation between the variables. I began with doing data ingestion, preprocessing, feature selection,hyperparameter tuning in order to improve perofrmance and best fit, visualization using scatterplot and deployment to git.

# In[173]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score


# In[174]:


df=pd.read_csv("salary_dataset.csv")
df


# In[175]:


df.head()


# In[176]:


#scatter plot
plt.scatter(df['YearsExperience'],df['Salary'])
plt.xlabel('YearsExperience')
plt.ylabel('Salary')


# In[177]:


##correlation
df.corr()


# In[178]:


##seaborn for visualization
import seaborn as sns
sns.pairplot(df)


# In[179]:


##independent and dependent features
X=df[['YearsExperience']]##independent feature
y=df[['Salary']]##dependent feature


# In[180]:


X_series=df['YearsExperience']

np.array(X_series).shape


# In[181]:


np.array(y).shape


# In[182]:


##Train Test Split
from sklearn.model_selection import train_test_split


# In[183]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


# In[184]:


X_train.shape


# In[185]:


##standardization
from sklearn.preprocessing import StandardScaler


# In[186]:


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)


# In[187]:


X_test=scaler.transform(X_test)


# In[188]:


X_test


# In[189]:


##Apply simple linear regression
from sklearn.linear_model import LinearRegression


# In[190]:


regression=LinearRegression(n_jobs=-1)


# In[191]:


regression.fit(X_train,y_train)


# In[192]:


print('coefficient of slope:',regression.coef_)
print('intercept:',regression.intercept_)


# In[193]:


#plot Training data plot for best fit line
plt.scatter(X_train,y_train)
plt.plot(X_train,regression.predict(X_train))


# In[194]:


##prediction for the test data
y_pred=regression.predict(X_test)


# In[195]:


##performance metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error 


# In[196]:


mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mse)
print(mse)
print(mae)
print(rmse)


# In[197]:


#In the context of a linear regression model, the Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) are metrics used to evaluate the performance of the model on a test dataset. Let's interpret the insights provided by these metrics:
#Mean Squared Error (MSE):
#Value: 38802588.99247059
#Interpretation: The MSE represents the average of the squared differences between the predicted and actual values. A lower MSE indicates better model performance. In this case, the MSE is relatively large, suggesting that the model's predictions have a considerable squared difference from the actual values.
#Mean Absolute Error (MAE):
#Value: 5056.9954666635895
#Interpretation: The MAE represents the average absolute difference between the predicted and actual values. Like MSE, lower values are desirable. In this case, the MAE is around 5056.99, indicating that, on average, the model's predictions are off by approximately this amount.
#Root Mean Squared Error (RMSE):
#Value: 6229.172416338352
#Interpretation: The RMSE is the square root of the MSE and provides a measure of the average magnitude of the errors in the same units as the target variable. As with MSE and MAE, a lower RMSE is better. The RMSE of approximately 6229.17 suggests that, on average, the model's predictions are off by around this amount.
#Overall Insight:
#The model's performance, based on these metrics, may be acceptable depending on the context of the problem. However, it's essential to compare these values to the scale of your target variable. For example, if predicting house prices, an RMSE of 6229.17 might be reasonable if the prices themselves are on a large scale.
#Consider comparing these metrics with the mean or median value of your target variable to put them into perspective. Additionally, it might be beneficial to compare the performance of this model with alternative models or different feature sets.
 


# In[198]:


from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print(score)


# In[200]:


#display adjusted R-Squared
1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)


# In[201]:


#The adjusted R2 score you've calculated is approximately 0.9238. This adjusted R2 takes into account both the number of predictors in your model and the sample size. A higher adjusted R2 generally indicates a better fit of the model to the data, considering the trade-off between model complexity (number of predictors) and performance.
#In your case, a value of 0.9238 suggests that the linear regression model, accounting for the number of predictors, explains about 92.38% of the variance in the target variable on the test set.
#It's important to interpret this value in the context of your specific problem and to consider it alongside other evaluation metrics and the goals of your analysis


# In[202]:


#OLS Linear Regression
import statsmodels.api as sm


# In[203]:


model=sm.OLS(y_train,X_train).fit()
prediction=model.predict(X_test)
print(prediction)


# In[204]:


#model: This represents the trained OLS linear regression model. You've fitted the model using the training data.
#prediction: This is an array of predicted values for the test set based on the trained model. Each value corresponds to the predicted response variable (target) for a specific set of predictor variables in the test set.
#It's important to note that in linear regression, predictions can sometimes be negative, and the interpretation of negative predictions depends on the context of your specific problem. If your target variable represents a quantity that cannot be negative (e.g., house prices), you may want to investigate why your model is predicting negative values.
#Additionally, you may want to evaluate the model's performance using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), �2R2 score, and adjusted �2R2 score, as we discussed earlier.


# In[205]:


print(model.summary())


# In[55]:


##prediction for new data
regression.predict(scaler.transform([[20]]))


# In[56]:


#scaler.transform([[20]]): This part of the code scales the input value 20 using the same scaler that was used to preprocess the training data. It's important to scale new data in the same way as the training data to ensure consistency in the model's predictions.
#regression.predict(): This part of the code uses the trained regression model (regression) to make predictions on the scaled input value.
#he output array([[211962.34742196]]) is the predicted output for the input value 20.


# In[ ]:




