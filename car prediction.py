#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv("car data.csv")


# In[3]:


df.head()


# In[4]:


# find the missing values
df.isnull().sum()


# In[5]:


df.shape


# In[6]:


# catogorical
print(df["Seller_Type"].unique())


# In[7]:


print(df["Transmission"].unique())


# In[8]:


print(df["Fuel_Type"].unique())
print(df["Owner"].unique())


# In[9]:


df.describe()


# In[10]:


df["Car_Name"].value_counts()


# In[11]:


# deal with year and avoid  car name
df.columns


# In[12]:


final_dataset=df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[13]:


final_dataset.head()


# In[14]:


final_dataset["current_year"]=2020


# In[15]:


final_dataset.head()


# In[16]:


final_dataset["no. of years"]=final_dataset["current_year"]-final_dataset["Year"]


# In[ ]:





# In[17]:


final_dataset.head()


# In[18]:


final_dataset.drop(["Year","current_year"],axis=1,inplace=True)


# In[19]:


final_dataset.head()


# In[20]:


from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()


# In[21]:


final_dataset["Fuel_Type"]=label.fit_transform(final_dataset["Fuel_Type"])
final_dataset["Seller_Type"]=label.fit_transform(final_dataset["Seller_Type"])
final_dataset["Transmission"]=label.fit_transform(final_dataset["Transmission"])


# In[22]:


Fuel_Type=final_dataset["Fuel_Type"]
Seller_Type=final_dataset["Seller_Type"]
Transmission=final_dataset["Transmission"]


# In[23]:


final_dataset.head()


# In[24]:


# or instead we can do like this
# final_dataset=pd.get_dummies(final_dataset,drop_first=True)


# In[25]:


final_dataset.corr()


# In[26]:


import seaborn as sns


# In[27]:


sns.pairplot(final_dataset)


# In[28]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

corrmat=final_dataset.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[29]:


final_dataset.head()


# In[30]:


# allocating x and y ie,independent and dependend features
x=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]
# or
# x=final_dataset.drop(["Selling_Price"],axis=1,inplace=True)
# y=final_dataset["Selling_Price"]


# In[31]:


x.head()


# In[32]:


y.head()


# In[33]:


# feature importance
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(x,y)


# In[35]:


print(model.feature_importances_)


# In[37]:


# plot graph of feature importances
feat_importance=pd.Series(model.feature_importances_,index=x.columns)
feat_importance.nlargest(5).plot(kind='barh')
plt.show()


# In[38]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# In[52]:


x_train.shape


# In[53]:


y_train.shape


# In[39]:


from sklearn.linear_model import LinearRegression


# In[40]:


model=LinearRegression()


# In[41]:


model.fit(x_train,y_train)


# In[42]:


predictions=model.predict(x_test)


# In[43]:


predictions


# In[44]:


model.score(x_test,y_test)


# In[45]:


sns.distplot(y_test-predictions)


# In[46]:


from sklearn import metrics


# In[49]:


print('MAE:',metrics.mean_absolute_error(y_test,predictions))
print('MSE:',metrics.mean_squared_error(y_test,predictions))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,predictions)))


# In[50]:


import pickle
# open a file, where you ant to store the data
file = open('Linear_regression_model_carprediction.pkl', 'wb')

# dump information to that file
pickle.dump(model, file)


# In[ ]:




