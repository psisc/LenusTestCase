import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

### Load data
data = pd.read_excel(r'Data_Scientist_-_Case_Dataset.xlsx')
df = pd.DataFrame(columns=['customer_id','converted','customer_segment','gender','age','related_customers','family_size','initial_fee_level','credit_account_id','branch'])
df[['customer_id','converted','customer_segment','gender','age','related_customers','family_size','initial_fee_level','credit_account_id','branch']] = data['customer_id,converted,customer_segment,gender,age,related_customers,family_size,initial_fee_level,credit_account_id,branch'].str.split(',',expand=True)

### Clean data
df.loc[df["age"]=="","age"] = "0.0" #Missing age input

df["age"] = df["age"].astype(float)
df["related_customers"] = df["related_customers"].astype(int)
df["family_size"] = df["family_size"].astype(int)
df["converted"] = df["converted"].astype(int)
df["initial_fee_level"] = df["initial_fee_level"].astype(float)
df["customer_segment"] = df["customer_segment"].astype(int)
df["has_credit_account"] = 1
df.loc[df["credit_account_id"]=="9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0","has_credit_account"] = 0

### Investigate distributions
plt.hist(df["converted"])  # density=False would make counts, bins=8,range=(0,200)
plt.ylabel('Counts')
plt.title('Converted distribution')
plt.xticks([0,1])
#plt.xlabel('Data')
plt.savefig(r'C:\Users\PerS1\OneDrive\Skrivebord\LenusCaseStudy\SimpleDataVisuals\ConvertedDistribution.png')
plt.show()

### Initial empty entries replaced with nan
df.loc[df["age"]==0,"age"] = np.nan
df.loc[df["branch"]=="","branch"] = np.nan

### Drop unnecessary columns
df.drop(["customer_id","credit_account_id"],axis=1,inplace=True)#Assuming specific credit account id has nothing to do with converted.


### Convert categorical columns into columns with 0 and 1
bran = pd.get_dummies(df["branch"])
gen = pd.get_dummies(df["gender"])
custseg = pd.get_dummies(df["customer_segment"])

custseg.columns = ["custseg_11","custseg_12","custseg_13"]

df2 = pd.concat([df.drop(['gender','branch','customer_segment'],axis=1),bran,gen,custseg],axis=1)

#Correlation Matrix with and without dropping nan rows.
corrM = df2.dropna().corr() 
corrM = df2.corr() 
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corrM, xticklabels=corrM.columns.values, yticklabels=corrM.columns.values, vmin=-1,vmax=1,cmap= 'RdBu_r',annot=True,ax=ax,fmt='.2f')
plt.savefig(r'C:\Users\PerS1\OneDrive\Skrivebord\LenusCaseStudy\CorrMatrixDropAgeNA.png')

### RFE
#Removing rows without an age entry, down to 714 rows, probably fine. Scale variables to zero mean and unit variance:
df3 = df2.dropna()
cols = ['age','related_customers','family_size','initial_fee_level','has_credit_account','Helsinki','Tampere','Turku','female','custseg_11','custseg_12']
X = df3[cols]
y = df3['converted']
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled,columns=X.columns)

model = LogisticRegression()
rfe = RFE(model,n_features_to_select=1)
fit = rfe.fit(X_scaled, y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
print(X.columns[fit.support_])

### Logistic regression using individual variables
for a in ['age','related_customers','family_size','initial_fee_level','female','has_credit_account']:
    x_train, x_test, y_train, y_test = train_test_split(X_scaled[a], y,random_state=42)
    model = LogisticRegression()

    model.fit(np.array(x_train).reshape(-1, 1), y_train)
    prediction = model.predict(np.array(x_test).reshape(-1,1))

    print(a)
    print(accuracy_score(y_test,prediction))
    print('--------')

x_train, x_test, y_train, y_test = train_test_split(X_scaled[['Helsinki','Tampere','Turku']], y,random_state=42)
model = LogisticRegression()

model.fit(x_train, y_train)
prediction = model.predict(x_test)
print('branch')
print(accuracy_score(y_test,prediction))
print('--------')

x_train, x_test, y_train, y_test = train_test_split(X_scaled[['custseg_11','custseg_12']], y,random_state=42)
model = LogisticRegression()

model.fit(x_train, y_train)
prediction = model.predict(x_test)
print('customer_segment')
print(accuracy_score(y_test,prediction))
print('--------')
    

### Multiple variables:
x_train, x_test, y_train, y_test = train_test_split(X_scaled[['female','has_credit_account','custseg_11','custseg_12','age']], y,random_state=42)
model = LogisticRegression()

model.fit(x_train, y_train)
prediction = model.predict(x_test)

print(accuracy_score(y_test,prediction))
# Confusion matrix
confusion_matrix(y_test, prediction)