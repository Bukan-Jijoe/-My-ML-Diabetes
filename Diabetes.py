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
CSV_PATH = os.path.join(os.getcwd(),'Dataset','diabetes.csv')

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

#%% 
# Normalization
model = LogisticRegression()
mms = MinMaxScaler()
ss = preprocessing.StandardScaler()
print(X_train.shape)

#%%
#Reshape the data 
if X_train.ndim == 1:
    X_train = X_train.reshape(-1,1)
if X_test.ndim == 1:
    X_test = X_test.reshape(-1,1)   

print(X_train.shape)
X_train_ss = ss.fit_transform(X_train)
X_test_ss = ss.transform(X_test)

model.fit(X_train_ss,y_train)



# %%
print(len(model.coef_))
print(model.coef_)
print(model.intercept_)
print(model.score(X_train_ss,y_train))



# %%

y_pred = model.predict(X_test_ss)

# %%
cr = classification_report(y_test, y_pred)
print(cr)

# %%
