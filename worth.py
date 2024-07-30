# Imports 
import sys
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy import stats
import statsmodels.api as sm
from pyspark.sql import SparkSession, functions, types
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
# data preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge # lin reg model
from sklearn.naive_bayes import GaussianNB # naive bayes classification model
from sklearn.neighbors import KNeighborsClassifier # kneighbors classification
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier # decision tree classifier 
from sklearn.ensemble import RandomForestClassifier # randomforest ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from pyspark.sql.functions import col, when
from sklearn.preprocessing import PolynomialFeatures
matplotlib.use('Agg')  # Use the Agg backend for non-GUI environments

# cost of making coffee at home per day

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
data_file = 'GACTT_RESULTS_ANONYMIZED_v2.csv'
# data = spark.read.csv(data_file, header=True, inferSchema=True)
data = pd.read_csv(data_file)
cup = data[["Submission ID", "How many cups of coffee do you typically drink per day?"]].dropna() #THIS WORKS
spent = data[["Submission ID", "In total, much money do you typically spend on coffee in a month?"]].dropna() #THIS WORKS
eq = data[["Submission ID", "Approximately how much have you spent on coffee equipment in the past 5 years?"]].dropna()

# inner join on submission ID drops people that didnt answer the question
join = pd.merge(cup, spent, how='inner', on='Submission ID')
join = pd.merge(join, eq, how='inner', on='Submission ID')

# transform cup drank per day into usable data
join['cup_assumptions'] = join['How many cups of coffee do you typically drink per day?'].apply(lambda x: "0" if x == "Less than 1" else x)
join['cup_assumptions'] = join['cup_assumptions'].apply(lambda x: "5" if x == "More than 4" else x)

# transform coffee spending per month into usable data
join['spent_assumptions'] = join['In total, much money do you typically spend on coffee in a month?'].apply(lambda x: "10" if x == "<$20" else x)
join['spent_assumptions'] = join['spent_assumptions'].apply(lambda x: "30" if x == "$20-$40" else x)
join['spent_assumptions'] = join['spent_assumptions'].apply(lambda x: "50" if x == "$40-$60" else x)
join['spent_assumptions'] = join['spent_assumptions'].apply(lambda x: "70" if x == "$60-$80" else x)
join['spent_assumptions'] = join['spent_assumptions'].apply(lambda x: "90" if x == "$80-$100" else x)
join['spent_assumptions'] = join['spent_assumptions'].apply(lambda x: "150" if x == ">$100" else x)

# monthly -> daily
join['spent_assumptions'] = join['spent_assumptions'].astype(int)
join['spent_assumptions'] = join['spent_assumptions'] / 30
join['spent_assumptions'] = join['spent_assumptions'].astype(str)

# More than $1,000 = $1250
# $500-$1000 = $750
# $300-$500 = $400
# $100-$300 = $200
# $50-$100 = $75
# $20-$50 = $35
# Less than $20 = $10
# transform spending on equipment into usable data
join['eq_assumptions'] = join['Approximately how much have you spent on coffee equipment in the past 5 years?'].apply(lambda x: "1250" if x == "More than $1,000" else x)
join['eq_assumptions'] = join['eq_assumptions'].apply(lambda x: "750" if x == "$500-$1000" else x)
join['eq_assumptions'] = join['eq_assumptions'].apply(lambda x: "400" if x == "$300-$500" else x)
join['eq_assumptions'] = join['eq_assumptions'].apply(lambda x: "200" if x == "$100-$300" else x)
join['eq_assumptions'] = join['eq_assumptions'].apply(lambda x: "75" if x == "$50-$100" else x)
join['eq_assumptions'] = join['eq_assumptions'].apply(lambda x: "35" if x == "$20-$50" else x)
join['eq_assumptions'] = join['eq_assumptions'].apply(lambda x: "10" if x == "Less than $20" else x)

brew_method_prices = {
    'Espresso': 500,  # Average price for a home espresso machine
    'Coffee brewing machine (e.g. Mr. Coffee)': 60,  # Average price for a basic coffee brewing machine
    'Coffee extract (e.g. Cometeer)': 0,  # Not a typical home brewing method with standard equipment
    'Other': 35,  # General price for unspecified methods
    'Pod/capsule machine (e.g. Keurig/Nespresso)': 150,  # Average price for a pod/capsule machine
    'Pour over': 40,  # Average price for a pour over setup
    'French press': 50,  # Average price for a French press
    'Cold brew': 40,  # Average price for a cold brew setup
    'Instant coffee': 0,  # Minimal equipment cost
    'Bean-to-cup machine': 1000,  # Average price for a bean-to-cup machine
}

# Function to calculate the total cost based on brewing methods
def calculate_brewing_cost(row, price_dict):
    methods = row['How do you brew coffee at home?'].split(', ')
    total_cost = sum(price_dict.get(method, 0) for method in methods)
    return total_cost

# Apply the function to the dataframe
join['brewing_cost'] = data.apply(lambda row: calculate_brewing_cost(row, brew_method_prices) if not pd.isna(row['How do you brew coffee at home?']) else 0, axis=1)


# join.drop(['How many cups of coffee do you typically drink per day?'], axis=1)
# cup assumption = how many cups of coffee you drink per day
# spent assumption = how much money you spend on coffee per day
# eq assumption = how much money do you spend on equipement per month
join = join[['Submission ID', 'cup_assumptions', 'spent_assumptions', 'eq_assumptions', 'brewing_cost']]
join.to_csv('worth.csv', index=False)

# cost of buying coffee per day

# 1. build a plot of
# X: time
# Y1: brewing at home costs (extra costs) * # cups daily (spent assumption)
# Y2: buy outisde price daily (cup assumption) * cups daily (spent assumption)
# not worth it (no) if Y2 never exceeds Y1 else worth it (yes)

join['cup_assumptions'] = pd.to_numeric(join['cup_assumptions'])
join['spent_assumptions'] = pd.to_numeric(join['spent_assumptions'])
join['eq_assumptions'] = pd.to_numeric(join['eq_assumptions'])
join['brewing_cost'] = pd.to_numeric(join['brewing_cost'])

# X vs Y1
y = []
for index, row in join.iterrows():
    cups_per_day = row['cup_assumptions']
    extra_costs = row['brewing_cost']
    cost = extra_costs * cups_per_day
    y.append(cost)
X = np.arange(1, len(y) + 1).reshape(-1,1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y)
model = KNeighborsRegressor(5)
model.fit(X_train, y_train)
print("Trainig R^2 is: ", model.score(X_train, y_train))
print("Validation R^2 is: ", model.score(X_valid, y_valid))

# Plotting the results
plt.figure(figsize=(14, 8))
plt.scatter(X, y, color='blue', alpha=0.5, label='Cost of brewing at home on that day')
plt.plot(X, model.predict(X), 'r-', label='line')
plt.xlabel('Sample')
plt.ylabel('Cost of Brewing Coffee at Home')
plt.title('Cost of Brewing coffee at home vs. Days (KNN Regression)')
plt.legend()
plt.savefig('worth_home.png', dpi=300)

# 2. build a neural network
# X: [eq assumptions, spent_assumptions, cost_assumptions]
# Y: yes (worth it to buy a coffee machine) or no (not worth it)