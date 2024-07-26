# Imports 
import sys
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


# join.drop(['How many cups of coffee do you typically drink per day?'], axis=1)
join = join[['Submission ID', 'cup_assumptions', 'spent_assumptions', 'eq_assumptions']]
join.to_csv('worth.csv', index=False)



# cost of buying coffee per day

# 1. build a plot of
#  X: time
# Y1: coffee machine (upfront cost one time) + extra * # cups daily * X
# Y2: buy outisde * num of cups * X
# not worth it (no) if Y2 never exceeds Y1 else worth it (yes)

# 2. build a neural network
# X: [eq assumptions, spent_assumptions, cost_assumptions]
# Y: yes (worth it to buy a coffee machine) or no (not worth it)