#%%
#Import
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes 
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer,IterativeImputer
import scipy.stats as ss
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


#%%
def cramers_corrected_stat(confusion_matrix):
    """"calculate Cramers V statistic for categorical-categorical data association"""
    chi2=ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0,phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k- ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1),(rcorr-1)))


# %%

os.chdir

CSV_PATH = os.path.join(os.getcwd(),'Dataset','diabetes.csv')

Model_path = os.path.join(os.getcwd(),'Model','model.pkl')

# %%
df = pd.read_csv(CSV_PATH)
df = df.drop("Age", axis = 1)
# %%
df.head()
# %%
df.describe() #finding the outliers
# %%
df.info() 
# %%
df
# %%
cat = ["Pregnancies","Outcome"] # cat is discrete data
con = df.drop(labels=cat, axis=1).columns # drop cat column, so only continous

# %%
# Plot for categorical data

for x in cat:
    plt.figure()
    sns.countplot(df[x])
    plt.show
# %%
# Plot for continuous data
for x in con:
    plt.figure()
    sns.distplot(df[x])
    plt.show

# There are 0 value that should not be there, as in Glucose, Bloodpressure, SkinThickness
# Insulin, BMI that require for further verification
# %%
df.boxplot()

# There are outliers exist in Insulin, BloodPressure,BMI, SkinThickness, Glucose,
# Pedigree, Age, Pregnancies
# to be concern: 


# %%
# how to check how many zero value in insulin column
print((df["Insulin"] == 0).sum()) # near  to half of the data is 0 (374)
print((df["BMI"] == 0).sum()) # data that is 0 (11)
print((df["SkinThickness"] == 0).sum()) # data that is 0 (227)
# %%
# to replace 0 with NANs

for x in con:
    df[x] = df[x].replace(0,np.nan)

df.isna().sum().sort_values(ascending = False)
df.duplicated().sum()

# %% Filling the NaN value 

columns_name = df.columns
knn = KNNImputer()
ii = IterativeImputer()
df = knn.fit_transform(df) # this function return as numpy array
df = pd.DataFrame(df) # to convert numpy array to pandas dataframe back
df.columns = columns_name

#%%
# to visualize the df after imputation
for i in con:
    plt.figure()
    sns.histplot(df[i], kde=True)
    plt.show

# %%
# Feature Selection - find the best features that correlates with target
# Continuous vs Categorical
# all the value times 100, to get value in % percentage, thus any below 50%
# we can decide whether to drop or remain the feature but above 50% considered necessary for model
for x in con:
    print(x)
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[x], axis=-1), df['Outcome'])
    print(lr.score(np.expand_dims(df[x], axis=-1), df['Outcome']))

# all the continuous data column is considerly important
# %%
# to find the correlation between categorical and categorical data
matrix = pd.crosstab(df['Pregnancies'], df["Outcome"]).to_numpy()
print(cramers_corrected_stat(matrix))

# it seems like the pregnancies column has less correlation with our predicted target
# %% DATA PREPROCESSING
print(con)
#con_dropped = con.drop(labels=["Age"])
#print(con_dropped)
X = df.loc[:,con]
y = df['Outcome']

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 42 )

#%% MODEL DEVELOPMENT -----> Pipeline

#Logistic Regression

pipeline_mms_lr = Pipeline([
    ("Min_Max_Scaler", MinMaxScaler()),
    ("Logisitic_Regression", LogisticRegression())
]) # Pipeline Steps

pipeline_ss_lr = Pipeline([
    ("Standard_Scaler", StandardScaler()),
    ("Logisitic_Regression", LogisticRegression())
]) # Pipeline Steps


#Decision Tree
pipeline_mms_dt = Pipeline([
    ("Min_Max_Scaler", MinMaxScaler()),
    ("Decision_Tree", DecisionTreeClassifier())
]) # Pipeline Steps

pipeline_ss_dt = Pipeline([
    ("Standard_Scaler", StandardScaler()),
    ("Decision_Tree", DecisionTreeClassifier())
]) # Pipeline Steps

#Random Forest Classifier

pipeline_mms_rf = Pipeline([
    ("Min_Max_Scaler", MinMaxScaler()),
    ("Random_Forest", RandomForestClassifier())
]) # Pipeline Steps

pipeline_ss_rf = Pipeline([
    ("Standard_Scaler", StandardScaler()),
    ("Random_Forest", RandomForestClassifier())
]) # Pipeline Steps

#KNN

pipeline_mms_knn = Pipeline([
    ("Min_Max_Scaler", MinMaxScaler()),
    ("KNN", KNeighborsClassifier())
]) # Pipeline Steps

pipeline_ss_knn = Pipeline([
    ("Standard_Scaler", StandardScaler()),
    ("KNN", KNeighborsClassifier())
]) # Pipeline Steps

#GradientBoost
pipeline_mms_gb = Pipeline([
    ("Min_Max_Scaler", MinMaxScaler()),
    ("Gradient_Boost", GradientBoostingClassifier())
]) # Pipeline Steps

pipeline_ss_gb = Pipeline([
    ("Standard_Scaler", StandardScaler()),
    ("Gradient_Boost", GradientBoostingClassifier())
]) # Pipeline Steps

#SVC

pipeline_mms_svc = Pipeline([
    ("Min_Max_Scaler", MinMaxScaler()),
    ("SVC", SVC())
]) # Pipeline Steps

pipeline_ss_svc = Pipeline([
    ("Standard_Scaler", StandardScaler()),
    ("SVC", SVC())
]) # Pipeline Steps


#a list to store all the pipelines

pipelines = [pipeline_mms_lr,pipeline_ss_lr,pipeline_mms_dt,pipeline_ss_dt,
             pipeline_mms_rf,pipeline_ss_rf,pipeline_mms_knn,pipeline_ss_knn,
             pipeline_mms_gb,pipeline_ss_gb,pipeline_mms_svc,pipeline_ss_svc]

for pipe in pipelines:
    pipe.fit(X_train,y_train)

pipe_score = []
for i,pipe in enumerate(pipelines):
    pipe_score.append(pipe.score(X_test,y_test))

print(pipelines[np.argmax(pipe_score)])
print(pipe_score[np.argmax(pipe_score)])

best_pipe = pipelines[np.argmax(pipe_score)]
# %% Classification Report
y_pred = best_pipe.predict(X_test)
cr = classification_report(y_test, y_pred)
print(cr)
# %% Hyperparameter Tuning & Grid Search CV

pipeline_mms_lr = Pipeline([
    ("Min_Max_Scaler", MinMaxScaler()),
    ("Logisitic_Regression", LogisticRegression())
]) # Pipeline Steps

# Hyperparameter Tuning
grid_param = [{'Logisitic_Regression__penalty':['l2',None],
               'Logisitic_Regression__C':[0.0001,0.001,0.01,0.1,1,10,100],
               'Logisitic_Regression__random_state':[1,42,123]
               }]

gridsearch = GridSearchCV(pipeline_mms_lr, grid_param, cv = 10, verbose = 1, n_jobs= -1)
grid = gridsearch.fit(X_train,y_train)
gridsearch.score(X_test, y_test)
print(grid.best_params_)
print(grid.best_estimator_)
print(grid.best_score_)

best_model = grid.best_estimator_
# %%
import pickle
with open (Model_path,"wb") as file:
    pickle.dump(best_model,file)
# %%
