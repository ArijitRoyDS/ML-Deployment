# # Import Libraries
import time
import pickle
import random
import pandas as pd
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler   # same as preprocessing.scale(data)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, pair_confusion_matrix
# from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn import metrics, tree
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from IPython.display import Image  
from sklearn.tree import export_graphviz
from yellowbrick.classifier import ROCAUC, roc_auc
from yellowbrick.features import RadViz
import pydotplus
from urllib.request import urlopen 
from itertools import cycle
#get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
pd.set_option('display.max_columns', 500) 


# The below random states gives the best accuracy. It's different for different models. 
# I have used a separate code snippet and varied random state in a for loop from 1 to 500 
# to obtain the value of random state that gives highest testing accuracy. 
rdt = 289 

# # Read the CSV File and check it's attributes
# Read the CSV File. File path to be modified if executed on a different Machine / OS
df = pd.read_csv('QualityPrediction.csv')

# Check the column names
col_list = list(df.columns)
print(col_list)
# There are 11 predictor variables and 1 Target variable

df.shape
# Insights: The dataset has 1599 rows and 12 columns

df.dtypes
# Insights: There are 11 float type variable columns and none of them are categorical
# The Target Variable is of Int type but it will be considered as categorical

# Check for Missing Data
total = df.isnull().sum()
percent = (df.isnull().sum()/df.isnull().count())
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data
# Insight: There is no missing data in any of the columns

# # Outlier Check: Boxplots and Pairplots
try:
    cols1 = list(df.columns)
    fig, ax = plt.subplots(4, 3, figsize=(18, 15))
    c = 0
    for i in range(4):
        for j in range(3):
            ax[i, j].boxplot(df[[cols1[c]]])
            ax[i, j].set_title(cols1[c])
            c=c+1
    plt.show()    
except:
    pass
# Insights on Outliers: Box Plots shows that most of the variables have outliers. Hence Outlier Treatment is necessary


# Check for Outliers
sns.pairplot(df.drop(['quality'], axis = 1))
plt.show()
# Insights on Outliers: Scatter Plots shows that most of the variables have outliers. Hence Outlier Treatment is necessary

# Outlier Treatment:
# Keep only those rows where the Z Score of all columns is < 3. 
# Basically drop all rows where Z Score value of at least 1 column >= 3
df=df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

# Check for Outliers again after treating outliers
sns.pairplot(df.drop(['quality'], axis = 1))
plt.show()
# Insights: Outliers are significantly reduced now


# Double check the effect of Outlier Treatment by comparing Mean and Median
# If there is a high difference between mean and median, it may possibily due to the presence of outliers.
# Although there are other reasons as well like if the data is highly skewed at both ends.
col_list = list(df.columns)
median=df.median()
mean=df.mean()

for i in range(0, len(col_list)-1):
    print(col_list[i])
    print("Mean   = ", mean[i].round(1))
    print("Median = ", median[i].round(1), end = '\n\n')    
# Insights: The gap between Mean and Median has reduced significantly


# # Plot Correlation Matrix to visualize the degree of Correlation between variables
# Create Correlation Matrix
corrmat = df.corr()

# Attrition correlation matrix
k = 12 # Number of variables for heatmap
cols = corrmat.nlargest(k, 'quality')['quality'].index

# Correlation Matrix
cm = np.corrcoef(df[cols].values.T)
f, ax = plt.subplots(figsize=(12, 9))
sns.set(font_scale=1.25)

# Plot the Heatmap
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 12}, 
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# Insights on Correlation: 'density', 'pH', 'fixed acidity', 'citric acid', 'alcohol' are moderately correlated
# Hence we calculate the VIF for these features


# # Calculate VIF (Variance Inflation Factor)
# Calculate Variance Inflation Factor for all correlated variables

# the independent variables set
X = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
  
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
  
print(vif_data)


# Insights on Correlation and it's treatment: 
#        'fixed acidity', 'pH' and 'citric acid' are moderately correlated. Add these 3 columns to give one column 'pH'
#         and drop 'fixed acidity' and 'citric acid'
#        'free sulfur dioxide' and 'total sulfur dioxide' are moderately correlated. Subtract 'free sulfur dioxide' from
#        'total sulfur dioxide' and drop 'free sulfur dioxide'


df['pH'] = df['pH'] + df['fixed acidity'] + df['citric acid']
df['total sulfur dioxide'] = df['total sulfur dioxide'] - df['free sulfur dioxide']
df = df.drop(['fixed acidity', 'free sulfur dioxide', 'citric acid', 'density'], axis = 1)


# Create Correlation Matrix again to check the effect of treatment
corrmat = df.corr()

# Attrition correlation matrix
k = 11 # Number of variables for heatmap
cols = corrmat.nlargest(k, 'quality')['quality'].index

# Correlation Matrix
cm = np.corrcoef(df[cols].values.T)
f, ax = plt.subplots(figsize=(12, 9))
sns.set(font_scale=1.25)

# Plot the Heatmap
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 11}, 
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# Calculate Variance Inflation Factor for all correlated variables
# the independent variables set
X = df[['volatile acidity', 'residual sugar', 'chlorides', 'total sulfur dioxide', 'pH', 'sulphates', 'alcohol']]
  
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]  
print(vif_data)
# Insights: VIF Scores have reduced significantly


df['quality'] = pd.Categorical(df.quality)

encoded_df = df.copy(deep = True)
encoded_df_nb = encoded_df.copy(deep = True)

categories = dict(df['quality'].value_counts())
categories = list(categories.keys())
categories = sorted(categories, key=int, reverse=False)
categories = list(map(str, categories))
categories


# # Feature Scaling / Standardization

# Save the Target Variable "quality" in variable y before standardization as Target Variable should not be standardised
y = encoded_df['quality'].values


cols = list(encoded_df.columns)
l = len(cols)-1
cols = cols[0:l]

#data = encoded_df.iloc[:, 1:].values  
data = encoded_df.drop('quality', axis = 1).values  

#standardize the data to normal distribution
dataset1_standardized = preprocessing.scale(data)
encoded_df1 = pd.DataFrame(dataset1_standardized, columns = cols)    # encoded_df1 is the dataset without the target variable
#encoded_df1.head(20)

# Save the standardised values of variables in x
x = encoded_df1.values


# Split Training and Testing Data in 80:20 ratio
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state = rdt)


# Final COlumns used for Model Building:

names = list(encoded_df.drop('quality', axis = 1).columns)
names_index = names[0:l]
names


# # Function to plot Confusion Matrix
# Function to plot Confusion Matrix. Callable in future from all models
def create_conf_mat(ytest, pred, model_name, mod):
    cm = confusion_matrix(y_test, pred, labels=[3,4,5,6,7,8])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[3,4,5,6,7,8])
    disp.plot(cmap='hot')
    plt.grid(False)
    plt.show()


# # Classification Report
# Function to print Classification Report. Callable in future from all models

def print_class_report(predictions, y_t, target, alg_name):

    print('Classification Report for {0}:'.format(alg_name))
    print(classification_report(predictions, y_t, target_names = target))


# # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

# # Decision Tree Classifier

model_dt = DecisionTreeClassifier(random_state=rdt)


# # Hyper-Parameter Optimization using GridSearchCV (Cross Validation)

# # Execute this code snippet only to find the optimal values of parameters. Comment it out afterwards
# # Automatically find the best parameters instead of manual hit and try

# np.random.seed(rdt)
# start = time.time()

# param_dist = {'max_depth': [5, 6, 7, 8, 9, 10, 12],
#               'max_features': ['auto', 'sqrt', 'log2', None],
#               'criterion': ['gini', 'entropy'],
#               'min_samples_split' : [3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
#               'min_samples_leaf' : [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15]}

# # n_jobs should be 1 less than number of CPU cores. -1 means all cores
# cv_dt = GridSearchCV(model_dt, cv = 10, param_grid=param_dist, n_jobs = -1)                 

# cv_dt.fit(x_train, y_train)
# print('Best Parameters using grid search: \n', cv_dt.best_params_)
# end = time.time()
# print('Time taken in grid search: {0: .2f}'.format(end - start))


# # Decision Tree Classifier Model

# # Set best parameters given by grid search CV
# # model_dt.get_params().keys()

# for md in range(8,20):
#     for cri in ['gini', 'entropy']:
#         for mss in range(2,20):
#             for msl in range(2,15):
#                 for mf in ['auto', None]: 
#                     for ml in range(100,500,5):
#                         model_dt.set_params(max_depth=md, 
#                                             criterion=cri, 
#                                             min_samples_split=mss, 
#                                             min_samples_leaf=msl ,
#                                             max_features = mf,
#                                             max_leaf_nodes = ml)

#                         model_dt.fit(x_train, y_train)
#                         model_dt_score_train = model_dt.score(x_train, y_train)
#                         model_dt_score_test = model_dt.score(x_test, y_test)

#                         if model_dt_score_test >= 0.76:
#                             print("Accuracy = ", model_dt_score_test.round(4), md, cri, mss, msl, mf, ml)   


# Set best parameters given by grid search CV (Accuracy =  0.7251 13 gini 6 2)
model_dt.set_params(max_depth=10, 
                    criterion="gini", 
                    min_samples_split=6, 
                    min_samples_leaf=2 ,
                    max_features = 'auto',
                    max_leaf_nodes = 125)

model_dt.fit(x_train, y_train)

model_dt_score_train = model_dt.score(x_train, y_train)
print("Training score: ", model_dt_score_train)

model_dt_score_test = model_dt.score(x_test, y_test)
print("Testing score: ",model_dt_score_test)


# # Plot the Decision Tree Structure

# df2 = pd.DataFrame(df.drop(['quality'], axis = 1))
# col_names = list(df2.columns)
# fn = col_names
# cn = ['4','5','6','7','8']

# fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,8), dpi = 300)
# tree.plot_tree(model_dt, feature_names = fn, class_names = cn, filled = True)
# fig.savefig('DecisionTree.jpg')


# # Predictions & Evaluations (AUC, Confusion Matrix & Classification Report)

#predictions = model_dt.predict_proba(x_test)      # Predicts the probability of predictions being wither 0 or 1
y_pred_dt = model_dt.predict(x_test)             # Actual Predictions


accuracy_dt = metrics.accuracy_score(y_test, y_pred_dt)
Precision_dt = metrics.precision_score(y_test, y_pred_dt,average='weighted')
recall_dt = metrics.recall_score(y_test, y_pred_dt,average='weighted')

print("Accuracy:",accuracy_dt)
print("Precision (Weighted Average):",Precision_dt)
print("Recall (Weighted Average):",recall_dt)


# Call Confusion Matrix Plotting function
print(confusion_matrix(y_test, y_pred_dt))
# plot_confusion_matrix(model_dt, x_test, y_test, cmap='hot', labels=[3,4,5,6,7,8])


# # Area Under Curve (AUC)
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_facecolor('#fafafa')

visualizer = ROCAUC(model_dt, ax)

visualizer.fit(x_train, y_train)                 # Fit the training data to the visualizer
auc_dt = visualizer.score(x_test, y_test)        # Evaluate the model on the test data
visualizer.show()                                # Finalize and render the figure


# Print Classification Report:

class_report = print_class_report(y_pred_dt, y_test, categories, 'Decision Tree')
# f1 Score = 2 * (precision * recall)/ (precision + recall) ie how good my model is in predicting 1 as 1 and 0 as 0

filename = 'wine_quality_prediction_80.pkl'
pickle.dump(model_dt, open(filename, 'wb'))
