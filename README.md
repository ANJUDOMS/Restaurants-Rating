# Restaurant Data Analysis

## Installation

1. Install the required dependencies: `pip install pandas numpy matplotlib scikit-learn`

## Usage
1. Import the necessary libraries:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
2. Load the data from the CSV file:
```python
df = pd.read_csv(r"C:\Users\USER\Documents\Restaurants.csv")
```
3. Preprocess the data:
   - Encode the categorical features as binary values.
   - Handle any missing values.
4. Analyze the data:
   - Calculate the percentage of restaurants that offer both table booking and online delivery.
   - Compare the average ratings of restaurants with and without table booking.
   - Analyze the availability of online delivery among restaurants with different price ranges.
   - Determine the most common price range and the average rating for each price range.
   - Identify the color that represents the highest average rating among different price ranges.
   - Extract additional features from the existing columns, such as the length of the restaurant name or address.

## API
The main functions used in the analysis are:
- `percentage(df)`: Calculates the percentage of restaurants that offer both table booking and online delivery.
- `comparison(df)`: Compares the average ratings of restaurants with and without table booking.
- `result = df.pivot_table(...)`: Analyzes the availability of online delivery among restaurants with different price ranges.
- `Common_Price_range = df.groupby('Price range').size().idxmax()`: Determines the most common price range.
- `df.groupby('Price range')["Aggregate rating"].mean()`: Calculates the average rating for each price range.
- `d = df.groupby("Rating color")["Aggregate rating"].mean().idxmax()`: Identifies the color that represents the highest average rating among different price ranges.
- `df['Restaurant Name Length'] = df['Restaurant Name'].str.len()`: Extracts the length of the restaurant name.
- LinearRegression(): Trains a linear regression model to predict the aggregate rating.
- DecisionTreeRegressor(): Trains a decision tree regression model to predict the aggregate rating
- RandomForestRegressor(): Trains a random forest regression model to predict the aggregate rating.
- mean_squared_error(): Calculates the mean squared error of the model predictions.
- r2_score(): Calculates the R-squared score of the model predictions.

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please create a new issue or submit a pull request.


## Testing
To test the performance of the different models, the code includes the following:

Splitting the data into training and testing sets.
Calculating the mean squared error (MSE) and R-squared (R^2) score for each model on the test set.
Comparing the MSE of the different models using a bar chart.
The results show that the Random Forest Regressor model has the lowest MSE and highest R^2 score, indicating it is the most accurate model for predicting the aggregate restaurant rating.
