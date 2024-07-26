# Imports 
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from pyspark.sql import SparkSession, functions, types
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
# data preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression # lin reg model
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

join = pd.merge(cup, spent, how='inner', on='Submission ID')
join['cup_assumptions'] = join['How many cups of coffee do you typically drink per day?'].apply(lambda x: "0" if x == "Less than 1" else x)
join['cup_assumptions'] = join['cup_assumptions'].apply(lambda x: "5" if x == "More than 4" else x)


join['spent_assumptions'] = join['In total, much money do you typically spend on coffee in a month?'].apply(lambda x: "10" if x == "<$20" else x)
join['spent_assumptions'] = join['spent_assumptions'].apply(lambda x: "30" if x == "$20-$40" else x)
join['spent_assumptions'] = join['spent_assumptions'].apply(lambda x: "50" if x == "$40-$60" else x)
join['spent_assumptions'] = join['spent_assumptions'].apply(lambda x: "70" if x == "$60-$80" else x)
join['spent_assumptions'] = join['spent_assumptions'].apply(lambda x: "90" if x == "$80-$100" else x)
join['spent_assumptions'] = join['spent_assumptions'].apply(lambda x: "150" if x == ">$100" else x)

# join.drop(['How many cups of coffee do you typically drink per day?'], axis=1)

join.to_csv('output.csv', index=False)



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