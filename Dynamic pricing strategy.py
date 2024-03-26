#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# In[2]:


df = pd.read_csv("dynamic_pricing.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df


# # Exploratory data Analysis

# In[6]:


#Descriptive statistics of data
df.describe()


# In[7]:


# plotting scatter plot between expected ride duration and historical cost of ride
fig = px.scatter(df, x ='Expected_Ride_Duration', y = 'Historical_Cost_of_Ride', 
                 title = 'Expected ride duration vs. Hostorical cost of ride', trendline ='ols')
fig.show()


# In[8]:


# distribution of historical cost of rides based on the vehicle type
fig = px.box(df, x='Vehicle_Type', y='Historical_Cost_of_Ride', title = 'Historical cost of ride distribution bt Vehicle type')
fig.show()


# In[9]:


#correlation matrix
corr_matrix = df.corr()
fig = go.Figure(data=go.Heatmap(z=corr_matrix.values,
                                x=corr_matrix.columns,
                                y=corr_matrix.columns,
                                colorscale= 'Viridis' ))
fig.update_layout(title='Correlation Matrix')
fig.show()


# ## implementing Dynamic Pricing Strategy

# In[10]:


import numpy as np


# In[11]:


#calculating demand multiplier based on percentile for high and low demand
high_demand_percentile =75
low_demand_percentile =25
df['demand_multiplier']=np.where(df['Number_of_Riders'] > np.percentile(df['Number_of_Riders'], high_demand_percentile),
                                 df['Number_of_Riders'] / np.percentile(df['Number_of_Riders'], high_demand_percentile),
                                 df['Number_of_Riders'] / np.percentile(df['Number_of_Riders'], low_demand_percentile))


# In[12]:


df


# In[13]:


#calculating supply multiplier based on percentile for highand low supply
high_supply_percentile = 75
low_supply_percentile = 25
df['supply_multiplier'] = np.where(df['Number_of_Drivers'] > np.percentile(df['Number_of_Drivers'], low_supply_percentile),
                                   np.percentile(df['Number_of_Drivers'], high_supply_percentile) / df['Number_of_Drivers'],
                                   np.percentile(df['Number_of_Drivers'], low_supply_percentile) / df['Number_of_Drivers'] )


# In[14]:


print(np.percentile(df['Number_of_Drivers'], low_supply_percentile), np.percentile(df['Number_of_Drivers'], high_supply_percentile))


# In[15]:


print(np.percentile(df['Number_of_Riders'], high_demand_percentile), np.percentile(df['Number_of_Riders'], low_demand_percentile))


# In[16]:


df


# In[17]:


# defining price adjustment factors for high and low demand/supply
demand_threshold_high = 1.2
demand_threshold_low = 0.8
supply_threshold_high = 0.8
supply_threshold_low = 1.2

#calculating adjusted ride cost for dynamic pricing
df['adjusted_ride_cost'] = df['Historical_Cost_of_Ride'] * (np.maximum(df['demand_multiplier'], demand_threshold_low) * 
np.maximum(df['supply_multiplier'], supply_threshold_high))


# In[18]:


df


# In[19]:


#calculate the profit percentage of each ride
df['profit percentage'] = ((df['adjusted_ride_cost'] - df['Historical_Cost_of_Ride'])/df['Historical_Cost_of_Ride'])*100

#identifying loss rides
loss_rides = df[df['profit percentage'] < 0]
loss_count = len(loss_rides)
#identifying profit rides
profit_rides = df[df['profit percentage'] >= 0]
profit_count = len(profit_rides)

#creating a donut chart to show the distribution of profitable and loss rides
labels=['Profitable rides', 'Loss rides']
values =[profit_count, loss_count]
fig = go.Figure(data = [go.Pie(labels= labels, values=values, hole=0.4)])
fig.update_layout(title='Profitability of Rides (Dynamic Pricing vs. Historical pricing)')
fig.show()


# In[20]:


fig = px.scatter(df, x='Expected_Ride_Duration', y='adjusted_ride_cost', title='Expected ride duration vs Cost of Ride',
                trendline='ols')
fig.show()


# In[21]:


from sklearn.preprocessing import StandardScaler


# In[22]:


def data_preprocessing_pipeline(df):
    # identifying numeric and categorical features
    numeric_features = df.select_dtypes(include=['float','int']).columns
    categorical_features = df.select_dtypes(include=['object']).columns
    # imputing missing values in numeric features
    df[numeric_features] = df[numeric_features].fillna(data[numeric_features].mean())
    # detecting and handling outliers in numeric features
    for feature in numeric_features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)
        df[feature] = np.where((df[feature] < lower_bound | df[feature] > upper_bound), df[feature].mean(), data[feature])
    
    # handling missing values in categorical features
    df[categorocal_features] = df[categorical_features].fillna(df[categorical_features].mode().iloc[0])
    
    return df


# In[23]:


# converting vehicle type into a numerical feature
df['Vehicle_Type'] = df['Vehicle_Type'].map({'Premium': 1, 'Economy':0})


# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


x = np.array(df[['Number_of_Riders', 'Number_of_Drivers', 'Vehicle_Type', 'Expected_Ride_Duration']])
y = np.array(df[['adjusted_ride_cost']])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

y_train = y_train.ravel()
y_test = y_test.ravel()

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(x_train, y_train)


# In[26]:


df.head()


# In[27]:


def get_vehicle_type_numeric(vehicle_type):
    vehicle_type_mapping = {'Premium': 1, 'Economy': 0}
    vehicle_type_numeric = vehicle_type_mapping.get(vehicle_type)
    return vehicle_type_numeric


# In[28]:


def predict_price(number_of_riders, number_of_drivers, vehicle_type, Expected_Ride_Duration):
    vehicle_type_numeric = get_vehicle_type_numeric(vehicle_type)
    if vehicle_type_numeric is None:
        raise ValueError('Invalid vehicle type')
    
    input_data = np.array([[number_of_riders, number_of_drivers, vehicle_type_numeric, Expected_Ride_Duration]])
    predicted_price=model.predict(input_data)
    
    return predicted_price


# In[29]:


predicted_price = predict_price(50, 25, 'Economy', 30)


# In[30]:


print("Predicted price:", predicted_price)


# In[31]:


ypred = model.predict(x_test)


# In[32]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test.flatten(), y=ypred, mode ='markers', name='Actual vs Predicted'))
fig.add_trace(go.Scatter(x=[min(y_test.flatten()), max(y_test.flatten())], 
                         y=[min(y_test.flatten()), max(y_test.flatten())],
                         mode ='lines',
                         name= 'Ideal',
                         line=dict(color='red', dash='dash')
                        ))
fig.update_layout(
    title = 'Actual vs Predicted values',
    xaxis_title = 'Actual Values',
    yaxis_title = 'Prediceted values',
    showlegend=True,
)
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:




