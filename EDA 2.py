#!/usr/bin/env python
# coding: utf-8

# ##Black Friday EDA and Feature Engineering
# 
# ##Cleaning and Preparing the data for model training

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[32]:


df_train = pd.read_csv('train.csv')
df_train.head()


# In[33]:


df_test = pd.read_csv('test.csv')
df_test.head()


# In[42]:


#merge both the datasets
df_final  = df_train.append(df_test)
df_final


# In[43]:


df_final.info()


# In[44]:


df_final.describe()


# In[45]:


#df.drop(['User_ID'],axis = 1)
df_final.head()


# In[46]:


pd.get_dummies(df_final['Gender'])


# In[39]:


#But in the approach above, we have to save above thing to somewhere and later have to replace in main dataset. Better will be when we use the below approach to do changes directly in the dataset


# In[47]:


df_final['Gender']= df_final['Gender'].map({'F':0,'M':1})


# In[48]:


df_final.head()


# In[49]:


## Handle categorical feature Age
df_final['Age'].unique()


# In[50]:


#pd.get_dummies(df['Age'],drop_first=True)
df_final['Age']=df_final['Age'].map({'0-17':1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7})


# In[30]:


##second technqiue
#from sklearn import preprocessing
 
# label_encoder object knows how to understand word labels.
#label_encoder = preprocessing.LabelEncoder()
 
# Encode labels in column 'species'.
#df['Age']= label_encoder.fit_transform(df['Age'])
 
#df['Age'].unique()


# In[51]:


df_final.head()


# In[52]:


df_final.drop(['User_ID'],axis = 1)


# In[53]:


df_final_city=pd.get_dummies(df['City_Category'],drop_first= True)


# In[54]:


df_final_city.head()


# In[55]:


df_final = pd.concat([df_final, df_final_city],axis=1)


# In[56]:


df_final.head()


# In[57]:


#Dropping City Category


# In[59]:


df_final.drop('City_Category', axis = 1,inplace=True)


# In[61]:


df_final.drop('User_ID', axis = 1, inplace =True)


# In[62]:


df_final.head()


# In[63]:


##Missing Values


# In[64]:


df_final.isnull().sum()


# In[65]:


#Focus on replacing missing values


# In[67]:


df['Product_Category_2'].unique()


# In[68]:


df_final['Product_Category_2'].value_counts()


# In[69]:


##Best way here is to replace missing values with mode


# In[73]:


df_final['Product_Category_2']=df_final['Product_Category_2'].fillna(df_final['Product_Category_2'].mode()[0])


# In[74]:


df_final['Product_Category_2'].isnull().sum()


# In[79]:


df_final['Product_Category_3']=df_final['Product_Category_3'].fillna(df_final['Product_Category_3'].mode()[0])


# In[80]:


df_final['Product_Category_3'].isnull().sum()


# In[84]:


df_final.head().isnull().sum()


# In[85]:


df_final.shape


# In[88]:


df_final['Stay_In_Current_City_Years'].unique()


# In[91]:


df_final['Stay_In_Current_City_Years']=df_final['Stay_In_Current_City_Years'].str.replace('+','')


# In[92]:


df_final


# In[93]:


df_final.info()


# In[94]:


##Convert objects into integers


# In[96]:


df_final['Stay_In_Current_City_Years']=df_final['Stay_In_Current_City_Years'].astype(int)
df_final.info()


# In[97]:


df_final['B']=df_final['B'].astype(int)


# In[98]:


df_final['C']=df_final['C'].astype(int)


# In[99]:


df_final.info()


# In[102]:


#sns.pairplot(df_final)


# In[103]:


##Visualisation Age vs Purchased
sns.barplot('Age','Purchase',hue='Gender',data=df_final)


# In[111]:


## Visualization of Purchase with occupation
sns.barplot('Occupation','Purchase',hue='Gender',data=df_final)


# In[112]:


## Visualization of Purchase with occupation
sns.barplot('Occupation','Purchase',hue='Gender',data=df_final)


# In[113]:


sns.barplot('Product_Category_2','Purchase',hue='Gender',data=df_final)


# In[117]:


sns.barplot('Product_Category_1','Purchase',hue='Gender',data=df_final)


# In[118]:


sns.barplot('Product_Category_3','Purchase', hue = 'Gender', data = df_final)


# In[120]:


df_final.head()


# In[124]:


###Feature Scaling
df_final_test = df_final[df_final['Purchase'].isnull()]


# In[126]:


df_final_train = df_final[~df_final['Purchase'].isnull()]


# In[137]:


X = df_final_train.drop('Purchase',axis=1)
X


# In[138]:


y= df_final_train['Purchase']


# In[139]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[145]:


X_train.drop('Product_ID', axis =1, inplace = True)


# In[146]:


X_test.drop('Product_ID', axis =1, inplace = True)


# In[147]:


##Feature Scaling


# In[148]:


from sklearn.preprocessing import StandardScaler


# In[149]:


sc = StandardScaler()


# In[150]:


X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[151]:


##Train The Model


# In[ ]:




