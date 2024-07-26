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

# spark = SparkSession.builder.appName('Coffee Analysis Project').getOrCreate()
# spark.sparkContext.setLogLevel('WARN')

# assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
# assert spark.version >= '3.2' # make sure we have Spark 3.2+


# data.printSchema()
# Schema
# survey_schema = types.StructType([
#     types.StructField('Submission ID', types.StringType()),
#     types.StructField('What is your age?', types.StringType()),
#     types.StructField('How many cups of coffee do you typically drink per day?', types.IntegerType()),
#     types.StructField('Where do you typically drink coffee?', types.StringType()),
#     types.StructField('Where do you typically drink coffee? (At home)', types.BooleanType()),
#     types.StructField('Where do you typically drink coffee? (At the office)', types.BooleanType()),
#     types.StructField('Where do you typically drink coffee? (On the go)', types.BooleanType()),
#     types.StructField('Where do you typically drink coffee? (At a cafe)', types.BooleanType()),
#     types.StructField('Where do you typically drink coffee? (Other)', types.BooleanType()),
#     types.StructField('How do you typically make your coffee?', types.StringType()),
#     types.StructField('How do you typically make your coffee? (Coffee maker)', types.BooleanType()),
#     types.StructField('How do you typically make your coffee? (Espresso machine)', types.BooleanType()),
#     types.StructField('How do you typically make your coffee? (Pour over)', types.BooleanType()),
#     types.StructField('How do you typically make your coffee? (French press)', types.BooleanType()),
#     types.StructField('How do you typically make your coffee? (AeroPress)', types.BooleanType()),
#     types.StructField('How do you typically make your coffee? (Instant coffee)', types.BooleanType()),
#     types.StructField('How do you typically make your coffee? (Cold brew)', types.BooleanType()),
#     types.StructField('How do you typically make your coffee? (Other)', types.BooleanType()),
#     types.StructField('What type of coffee do you typically drink?', types.StringType()),
#     types.StructField('What type of coffee do you typically drink? (Regular)', types.BooleanType()),
#     types.StructField('What type of coffee do you typically drink? (Decaf)', types.BooleanType()),
#     types.StructField('What type of coffee do you typically drink? (Espresso)', types.BooleanType()),
#     types.StructField('What type of coffee do you typically drink? (Cold brew)', types.BooleanType()),
#     types.StructField('What type of coffee do you typically drink? (Latte)', types.BooleanType()),
#     types.StructField('What type of coffee do you typically drink? (Cappuccino)', types.BooleanType()),
#     types.StructField('What type of coffee do you typically drink? (Mocha)', types.BooleanType()),
#     types.StructField('What type of coffee do you typically drink? (Macchiato)', types.BooleanType()),
#     types.StructField('What type of coffee do you typically drink? (Other)', types.BooleanType()),
#     types.StructField('How do you typically drink your coffee?', types.StringType()),
#     types.StructField('How do you typically drink your coffee? (Black)', types.BooleanType()),
#     types.StructField('How do you typically drink your coffee? (With milk or cream)', types.BooleanType()),
#     types.StructField('How do you typically drink your coffee? (With sugar or sweetener)', types.BooleanType()),
#     types.StructField('How do you typically drink your coffee? (With flavor syrup)', types.BooleanType()),
#     types.StructField('How do you typically drink your coffee? (Other)', types.BooleanType()),
#     types.StructField('How important is coffee to your daily routine?', types.StringType()),
#     types.StructField('Do you have a preferred coffee brand?', types.BooleanType()),
#     types.StructField('What is your preferred coffee brand?', types.StringType()),
#     types.StructField('How often do you buy coffee from a coffee shop?', types.StringType()),
#     types.StructField('How much do you typically spend on coffee per week?', types.FloatType()),
#     types.StructField('Gender', types.StringType()),
#     types.StructField('Gender (please specify)', types.StringType()),
#     types.StructField('Education Level', types.StringType()),
#     types.StructField('Ethnicity/Race', types.StringType()),
#     types.StructField('Ethnicity/Race (please specify)', types.StringType()),
#     types.StructField('Employment Status', types.StringType()),
#     types.StructField('Number of Children', types.IntegerType()),
#     types.StructField('Political Affiliation', types.StringType())
# ])

# survey_schema = types.StructType([
#     types.StructField('Submission ID', types.StringType()),
#     types.StructField('What is your age?', types.StringType()),
#     types.StructField('How many cups of coffee do you typically drink per day?', types.StringType()),
#     types.StructField('Where do you typically drink coffee?', types.StringType()),
#     types.StructField('Where do you typically drink coffee? (At home)', types.StringType()),
#     types.StructField('Where do you typically drink coffee? (At the office)', types.StringType()),
#     types.StructField('Where do you typically drink coffee? (On the go)', types.StringType()),
#     types.StructField('Where do you typically drink coffee? (At a cafe)', types.StringType()),
#     types.StructField('Where do you typically drink coffee? (Other)', types.StringType()),
#     types.StructField('How do you typically make your coffee?', types.StringType()),
#     types.StructField('How do you typically make your coffee? (Coffee maker)', types.StringType()),
#     types.StructField('How do you typically make your coffee? (Espresso machine)', types.StringType()),
#     types.StructField('How do you typically make your coffee? (Pour over)', types.StringType()),
#     types.StructField('How do you typically make your coffee? (French press)', types.StringType()),
#     types.StructField('How do you typically make your coffee? (AeroPress)', types.StringType()),
#     types.StructField('How do you typically make your coffee? (Instant coffee)', types.StringType()),
#     types.StructField('How do you typically make your coffee? (Cold brew)', types.StringType()),
#     types.StructField('How do you typically make your coffee? (Other)', types.StringType()),
#     types.StructField('What type of coffee do you typically drink?', types.StringType()),
#     types.StructField('What type of coffee do you typically drink? (Regular)', types.StringType()),
#     types.StructField('What type of coffee do you typically drink? (Decaf)', types.StringType()),
#     types.StructField('What type of coffee do you typically drink? (Espresso)', types.StringType()),
#     types.StructField('What type of coffee do you typically drink? (Cold brew)', types.StringType()),
#     types.StructField('What type of coffee do you typically drink? (Latte)', types.StringType()),
#     types.StructField('What type of coffee do you typically drink? (Cappuccino)', types.StringType()),
#     types.StructField('What type of coffee do you typically drink? (Mocha)', types.StringType()),
#     types.StructField('What type of coffee do you typically drink? (Macchiato)', types.StringType()),
#     types.StructField('What type of coffee do you typically drink? (Other)', types.StringType()),
#     types.StructField('How do you typically drink your coffee?', types.StringType()),
#     types.StructField('How do you typically drink your coffee? (Black)', types.StringType()),
#     types.StructField('How do you typically drink your coffee? (With milk or cream)', types.StringType()),
#     types.StructField('How do you typically drink your coffee? (With sugar or sweetener)', types.StringType()),
#     types.StructField('How do you typically drink your coffee? (With flavor syrup)', types.StringType()),
#     types.StructField('How do you typically drink your coffee? (Other)', types.StringType()),
#     types.StructField('How important is coffee to your daily routine?', types.StringType()),
#     types.StructField('Do you have a preferred coffee brand?', types.StringType()),
#     types.StructField('What is your preferred coffee brand?', types.StringType()),
#     types.StructField('How often do you buy coffee from a coffee shop?', types.StringType()),
#     types.StructField('How much do you typically spend on coffee per week?', types.StringType()),
#     types.StructField('Gender', types.StringType()),
#     types.StructField('Gender (please specify)', types.StringType()),
#     types.StructField('Education Level', types.StringType()),
#     types.StructField('Ethnicity/Race', types.StringType()),
#     types.StructField('Ethnicity/Race (please specify)', types.StringType()),
#     types.StructField('Employment Status', types.StringType()),
#     types.StructField('Number of Children', types.StringType()),
#     types.StructField('Political Affiliation', types.StringType())
# ])

# Collecting data
# np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
data_file = 'GACTT_RESULTS_ANONYMIZED_v2.csv'
# data = spark.read.csv(data_file, header=True, inferSchema=True)
data = pd.read_csv(data_file)
cup = data[["Submission ID", "How many cups of coffee do you typically drink per day?"]].dropna() #THIS WORKS
spent = data[["Submission ID", "In total, much money do you typically spend on coffee in a month?"]].dropna() #THIS WORKS

# inner join on submission ID drops people that didnt answer the question
join = pd.merge(cup, spent, how='inner', on='Submission ID')

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

# join.drop(['How many cups of coffee do you typically drink per day?'], axis=1)
join = join[['Submission ID', 'cup_assumptions', 'spent_assumptions']]
join.to_csv('correlation.csv', index=False)
# join1.to_csv('output1.csv', index=False)

# x:  spent_assumptions
# y:  cup_assumptions
# setup ML pipeline
model = make_pipeline(
    SimpleImputer(strategy='mean'),  # Impute missing values
    MinMaxScaler(),  # Scale features to 0-1
    PolynomialFeatures(degree=2, include_bias=True),  # Create polynomial features
    Ridge(alpha=1.0)  # Apply Ridge regression with regularization
)

x = join['spent_assumptions'].values.astype(float)  # Ensure values are float
y = join['cup_assumptions'].values.astype(float)  # Ensure values are float
X = np.stack([x], axis=1)
X_train, X_valid, y_train, y_valid \
    = train_test_split(X, y) # create training and validation data 75/25 split

model.fit(X_train, y_train) # applies pipeline model to this data
ridge = model.named_steps['ridge']
coefs = ridge.coef_
intercept = ridge.intercept_

print("Coefficients: ", coefs)
print("Intercept: ", intercept)  # Note: intercept is zero because fit_intercept=False
print("Trainig R^2 is: ", model.score(X_train, y_train))
print("Validation R^2 is: ", model.score(X_valid, y_valid))
print("Trainig R is: ", math.sqrt(model.score(X_train, y_train)))
print("Validation R is: ", math.sqrt(model.score(X_valid, y_valid)))

# generate random X values for testing
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1) 
plt.plot(X_range[:, 0], model.predict(X_range), 'r-')

plt.figure(figsize=(10, 6))

plt.scatter(x, y, color='blue', alpha=0.5, label='Data Points')
plt.plot(X_range[:, 0], model.predict(X_range), 'r-', label='Polynomial Regression')
plt.title('Polynomial Regression of Cup Assumptions vs. Spent Assumptions')
plt.xlabel('Spent Assumptions')
plt.ylabel('Cup Assumptions')
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig('spent_vs_cup_assumptions_plot.png', dpi=300)

plt.figure(figsize=(10, 6))

jitter = 0.1 * np.random.randn(len(y)) # jitter y values 

plt.scatter(x, y + jitter, color='blue', alpha=0.5, label='Data Points (jittered)')
plt.plot(X_range[:, 0], model.predict(X_range), 'r-', label='Polynomial Regression')
plt.title('Polynomial Regression of Cup Assumptions vs. Spent Assumptions')
plt.xlabel('Spent Assumptions')
plt.ylabel('Cup Assumptions')
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig('spent_vs_cup_assumptions_jitter_plot.png', dpi=300)



# data.printSchema()

# Extract data

# spent = data.filter(data['In total, much money do you typically spend on coffee in a month?'] =
# ).dropna()
# test_spent = data.select(data['In total, much money do you typically spend on coffee in a month?'])

# join = data.select(data['How many cups of coffee do you typically drink per day?'], data['In total, much money do you typically spend on coffee in a month?']).dropna()
# join.show(4042, truncate=False)


# spent = data.select(data["Submission ID"])
# # spent.show(200)

# cups = data.select(data["Submission ID"], data['How many cups of coffee do you typically drink per day?']).dropna()
# cups.show(4042, truncate=False)

# cups.show()

# join.show()

# filter
# Assumption: take center of range 
# cleaned_spent = spent.withColumn("assumption", spent["In total, much money do you typically spend on coffee in a month?"])
# # cleaned_spent.show()
# cleaned_spent = cleaned_spent.withColumn(
#     "assumption",
#     when(col("assumption") == "<$20", "10").otherwise(col("assumption")),
# )
# cleaned_spent = cleaned_spent.withColumn(
#     "assumption",
#     when(col("assumption") == "$20-$40", "30").otherwise(col("assumption"))
# )
# cleaned_spent = cleaned_spent.withColumn(
#     "assumption",
#     when(col("assumption") == "$40-$60", "50").otherwise(col("assumption"))
# )
# cleaned_spent = cleaned_spent.withColumn(
#     "assumption",
#     when(col("assumption") == "$60-$80", "70").otherwise(col("assumption"))
# )
# cleaned_spent = cleaned_spent.withColumn(
#     "assumption",
#     when(col("assumption") == "$80-$100", "90").otherwise(col("assumption"))
# )
# cleaned_spent = cleaned_spent.withColumn(
#     "assumption",
#     when(col("assumption") == ">$100", "150").otherwise(col("assumption"))
# )

# # cleaned_spent.show()

# # Assumption: less than 1 = 0, More than 4 = 5
# cleaned_cups = cups.withColumn("assumption", cups["How many cups of coffee do you typically drink per day?"])
# # cleaned_cups.show()
# cleaned_cups = cleaned_cups.withColumn(
#     "assumption",
#     when(col("assumption") == "Less than 1", "0").otherwise(col("assumption")),
# )
# cleaned_cups = cleaned_cups.withColumn(
#     "assumption",
#     when(col("assumption") == "More than 4", "5").otherwise(col("assumption"))
# )

# cleaned_cups.show()

# linear reg

# Show cleaned data
# print(pandas_df.head(100))














# for col in data.columns:
#     print(col)

# cleaning

# convert null to 0, less than 1 to 0

# filter select

# Predict 