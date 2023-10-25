#!/usr/bin/env python
# coding: utf-8

# In[3]:


# import required libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#load the dataset
data=pd.read_csv("C:\CC\Salary.csv")


# In[4]:


train_data,test_data,train_labels,test_labels=train_test_split(data.drop("Salary",axis=1),data["Salary"],test_size=0.2,random_state=42)


# In[5]:


#train the model
lin_reg=LinearRegression()
lin_reg.fit(train_data,train_labels)
#predict the salary on the test data
pred_labels=lin_reg.predict(test_data)


# In[6]:


predicted_salary=np.array("pred_labels")


# In[7]:


pred_labels


# In[8]:


actual_salary=np.array(data["Salary"])


# In[9]:


actual_salary


# In[10]:


#plot a graph
plt.plot(actual_salary,label='Actual Salary')
plt.plot(pred_labels,label='Predicted Salary')
# add axis labels and title
plt.ylabel('Predicted Salary')
plt.xlabel('Actual Salary')
plt.title('Actual Vs Predicted Salaries')
#Add legend and grid
plt.legend()
plt.grid(True)
# display the plot
plt.show()


# In[13]:


print("Training set score:{:.3f}".format(lin_reg.score(train_data,train_labels)))
print("Test set score:{:.3f}".format(lin_reg.score(test_data,test_labels)))


# In[ ]:




