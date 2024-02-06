import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

file_path = 'vehicles_us.csv'
car_data = pd.read_csv(file_path)

st.header('Software Development Tools: Project')
st.header('_Vehicles_ US :green[Project] :🧘‍♀️:')

car_data['date_posted'] = pd.to_datetime(car_data['date_posted'])

# Select only numeric columns for resampling
numeric_cols = car_data.select_dtypes(include=[np.number])

# Add 'date_posted' to numeric_cols for resampling
numeric_cols['date_posted'] = car_data['date_posted']

# Resample numeric data on a monthly basis at the end of the month
monthly_data = numeric_cols.resample('M', on='date_posted').mean()

# Time series plot for average price
plt.figure(figsize=(12, 6))
plt.plot(monthly_data.index, monthly_data['price'], marker='o')
plt.title('Monthly Average Car Price Over Time')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.grid(True)
plt.show()

# Check if there's enough data for seasonal decomposition
if len(monthly_data) >= 24:  # Typically need at least 2 years of data
    decomposed = seasonal_decompose(monthly_data['price'].dropna(), model='additive')
    plt.figure(figsize=(12, 8))
    decomposed.plot()
    plt.show()


show_popularity_chart = st.checkbox("Show Popularity of Car Types")
show_price_chart = st.checkbox("Show Average Price by Car Type")

# Create the popularity chart conditionally
if show_popularity_chart:
    car_type_counts = car_data['type'].value_counts()
    plt.figure(figsize=(10, 5))
    car_type_counts.plot(kind='bar')
    plt.title('Popularity of Car Types')
    plt.xlabel('Type')
    plt.ylabel('Number of Listings')
    st.pyplot()

# Create the average price chart conditionally
if show_price_chart:
    average_price_by_type = car_data.groupby('type')['price'].mean()
    plt.figure(figsize=(10, 5))
    average_price_by_type.plot(kind='bar')
    plt.title('Average Price by Car Type')
    plt.xlabel('Type')
    plt.ylabel('Average Price')
    st.pyplot()


# Create the checkbox with a clear label
show_regression_line = st.checkbox("Show Regression Line")

# Create the scatter plot
plt.figure(figsize=(8, 6))  
plt.scatter(car_data['odometer'], car_data['price'])
plt.title('Car Price vs. Odometer Reading')
plt.xlabel('Odometer')
plt.ylabel('Price')

# Add the regression line conditionally
if show_regression_line:
    m, b = np.polyfit(car_data['odometer'], car_data['price'], 1)  # Calculate regression coefficients
    plt.plot(car_data['odometer'], m * car_data['odometer'] + b, color='red', label='Regression Line')
    plt.legend()  # Display the legend if a line is added

# Display the chart using st.pyplot
st.pyplot()
