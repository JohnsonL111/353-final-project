# CMPT353 - Final Report

## Purpose

There are two (2) main Python files for the analysis of data from Youtube Video [James Hoffman's Great American Taste Test](https://www.youtube.com/watch?v=bMOOQfeloH0&ab_channel=JamesHoffmann).

### correlation.py

This file contains the entire ETL process, ML pipeline, and plots using matplotlib for our correlation analysis between the number of cups of coffee vs. money spent on coffee daily. It will print out several p-values, R-values, and experimental debug prints and also create the plots spent_vs_cup_assumptions_plot.png, spent_vs_cup_assumptions_jitter_plot.png, and correlation.csv which contains our post-ETL data.

### worth.py

This file contains the entire ETL process, neural network classification model, accuracy scores, and inferential statistical test calculations for our coffee classification results. It will print R-values, classifcation model test accuracy, and inferential test results. It will also produce 2 scatter plots with best fit line: plots worth_home.png and worth_outside.png and a histogram titled final_result.png which contains the final results of our analysis. Lastly, it produces worth.csv which contains our post-ETL data.

## Order of operation

Both .py files (correlation.py, worth.py) are independent of each other, so they could be run in any order.

## Commands

Installing Python Packages
`$ pip install -r requirements.txt`

Running correlation.py

`$ python correlation.py`

Running worth.py

`$ python worth.py`

## Expected Output

Outputs and results of a successful execution of `correlation.py` and `worth.py`

- correlation.csv - CSV after data cleaning and merging
- spent_vs_cup_assumptions_plot.png - Scatter Plot with Polynomial Regression
- spent_vs_cup_assumptions_jitter_plot.png. - Scatter Plot with Jittered Data Points and Polynomial Regression
- worth.csv - CSV after data cleaning and merging
- worth_home.png - Plots the Cost of Brewing Coffee
- worth_outside.png - Plot the Cost of Buying Coffee Outside
- final_result.png - Histogram Plot of Comparing buying outside vs brewing at home

## Libraries Used

- numpy
- pandas
- scipy
- matplotlib
- sklearn
- pyspark
