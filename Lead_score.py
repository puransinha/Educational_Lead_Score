# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 22:55:07 2020

@author: Puran Prakash Sinha
"""
'''
An education company named X Education sells online courses to industry professionals.
On any given day, many professionals who are interested in the courses land on 
their website and browse for courses.
The company markets its courses on several websites and search engines like Google.
Once these people land on the website, they might browse the courses or fill up 
a form for the course or watch some videos. 
When these people fill up a form providing their email address or phone number, 
they are classified to be a lead. Moreover, the company also gets leads through 
past referrals. Once these leads are acquired, employees from the sales team
start making calls, writing emails, etc. 
Through this process, some of the leads get converted while most do not. 
The typical lead conversion rate at X education is around 30%.
Now, although X Education gets a lot of leads, its lead conversion rate is very poor.
For example, if, say, they acquire 100 leads in a day, only about 30 of them 
are converted. To make this process more efficient, the company wishes to 
identify the most potential leads, also known as ‘Hot Leads’. 
If they successfully identify this set of leads, the lead conversion rate 
should go up as the sales team will now be focusing more on communicating 
with the potential leads rather than making calls to everyone.
There are a lot of leads generated in the initial stage but only a few 
of them come out as paying customers at the last stage. 
In the middle stage, you need to nurture the potential leads well 
(i.e. educating the leads about the product, constantly communicating etc. ) 
in order to get a higher lead conversion.
X Education has appointed you to help them select the most promising leads, 
i.e. the leads that are most likely to convert into paying customers. 
The company requires you to build a model wherein you need to assign a lead 
score to each of the leads such that the customers with higher lead score 
have a higher conversion chance and the customers with lower lead score have
a lower conversion chance.


'''
# Import the relevant Libraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style='ticks')
import warnings
warnings.filterwarnings('ignore')

# Reading the Datasets
df=pd.read_csv(r'D:\Assignments\Lead_score\LeadsXEducation.csv')
df.head()

# check duplicates if yes.. remove duplicates...

dup=df.duplicated('Prospect ID').sum()
dup

sum(df.duplicated(subset='Prospect ID'))==0

## Inspection of Data sets

df.shape
df.info()
df.describe()
df.isnull().sum(axis=1)
df.isnull().sum(axis=0)
len(df)

# Replace the 'Select' with NaN values
df.replace('Select', np.nan)
df.Converted.value_counts()


#Calculating the null %ages 
round((df.isnull().sum()/len(df))*100,2)

# Drop the Null values which is more than 70%
df = df.drop(df.loc[:,list(round(100*(df.isnull().sum()/len(df.index)), 2)>70)].columns, 1)

# Checking all the Columns one by one
df.columns
df['Lead Quality'].describe()
sns.countplot(df['Lead Quality'])

df['Lead Quality']=df['Lead Quality'].replace(np.nan, 'Unknown')
sns.countplot(df['Lead Quality'])

# Dropping the irrelevant columns
df.drop(['Asymmetrique Activity Index','Asymmetrique Profile Index'], axis=1, inplace=True)
df

# filling the NaN values with 'Unknown' after verifying 
df.columns
df['Asymmetrique Activity Score'].value_counts()

df['Asymmetrique Activity Score'].isnull().sum()

df['Asymmetrique Activity Score']=df['Asymmetrique Activity Score'].fillna('Unknown')
df['Asymmetrique Activity Score']

df['Asymmetrique Profile Score']=df['Asymmetrique Profile Score'].fillna('Unknown')
df['City']=df['City'].fillna('Unknown')

df['Lead Profile']= df['Lead Profile'].fillna('Unknown')
df['Tags']=df['Tags'].fillna('Unknown')
df['What is your current occupation']= df['What is your current occupation'].fillna('Unknown')

df['Specialization']=df['Specialization'].fillna('Unknown')
df['Specialization'].value_counts()
df['Specialization'].replace('Select', 'Unknown', inplace=True)

df['Last Activity'].value_counts()
df['Last Activity']=df['Last Activity'].fillna('Unknown')

df['TotalVisits'].value_counts()
df['TotalVisits'].replace(np.NaN, df['TotalVisits'].median(), inplace=True)

df['Page Views Per Visit'].value_counts()
df['Page Views Per Visit'].replace(np.NaN, df['Page Views Per Visit'].median(), inplace=True)

df[['Country','City']]

df['Country'].value_counts()
df['City'].value_counts()
df['City'].replace('Select', 'Unknown', inplace=True)

sum(df['Country']=='India')/len(df.index)

df['Country']=df['Country'].apply(lambda x:'India' if x=='India' else 'Foreign Country')
df['Country'].value_counts()


df['Country']=df['Country'].replace('Select', 'Unknown')
df.isnull().sum()

df.drop(['How did you hear about X Education','What matters most to you in choosing a course' ], axis=1, inplace=True)

df['Lead Source'].value_counts
df['Lead Source'].replace(np.NaN, 'Unknown', inplace=True)

#checking the %age of null value
(df.isnull().sum())/len(df.index)

df.shape
df.info()
df.describe()

df.dropna()
df.columns

for i in df.columns:
    print(i, ':', df[i].nunique())

# Dropping the irrelevant columns 
df.drop(['Prospect ID'], axis=1, inplace=True)
df.drop(['Magazine', 'Receive More Updates About Our Courses',
         'Update me on Supply Chain Content', 'Get updates on DM Content',
         'I agree to pay the amount through cheque' ], axis=1, inplace=True)

df.head(10)

# Mapping the Numerical Values 1 and 0 to convert the Object to numeric
def mapp(x):
    return x.map({'Yes':1, 'No':0})

list_col=['Do Not Email', 'Do Not Call', 'Search', 'Newspaper Article',
          'X Education Forums', 'Newspaper', 'Digital Advertisement',
          'Through Recommendations', 'A free copy of Mastering The Interview']
df[list_col]=df[list_col].apply(mapp)

df.head()
df.columns
df.info()

# Converting all the Categorical columns with Numerical values using Get_dummies
df = pd.get_dummies(df, columns=['Lead Origin', 'Lead Source', 'Country',
                                 'Last Notable Activity'], drop_first=True)


dummy=pd.get_dummies(df['Asymmetrique Activity Score'], prefix='Asymmetrique Activity Score')
dummy
final_dummy=dummy.drop(['Asymmetrique Activity Score_Unknown'],1)
df=pd.concat([df,final_dummy], axis=1)


dummy=pd.get_dummies(df['Asymmetrique Profile Score'], prefix='Asymmetrique Profile Score')
final_dummy=dummy.drop(['Asymmetrique Profile Score_Unknown'], 1)
df=pd.concat([df,final_dummy], axis=1)

dummy=pd.get_dummies(df['Last Activity'],prefix='Last Activity')
final_dummy=dummy.drop(['Last Activity_Unknown'], 1)
df=pd.concat([df,final_dummy], axis=1)

dummy=pd.get_dummies(df['What is your current occupation'],prefix='What is your current occupation')
final_dummy=dummy.drop(['What is your current occupation_Unknown'], 1)
df=pd.concat([df,final_dummy], axis=1)

dummy=pd.get_dummies(df['Lead Profile'],prefix='Lead Profile')
final_dummy=dummy.drop(['Lead Profile_Unknown'], 1)
df=pd.concat([df,final_dummy], axis=1)

dummy=pd.get_dummies(df['Specialization'],prefix='Specialization')
final_dummy=dummy.drop(['Specialization_Unknown'], 1)
df=pd.concat([df,final_dummy], axis=1)

dummy=pd.get_dummies(df['City'], prefix='City')
final_dummy=dummy.drop(['City_Unknown'], 1)
df=pd.concat([df,final_dummy], axis=1)

dummy = pd.get_dummies(df['Lead Quality'], prefix='Lead Quality')
final_dummy = dummy.drop(['Lead Quality_Unknown'], 1)
df = pd.concat([df,final_dummy], axis=1)

dummy=pd.get_dummies(df['Tags'],prefix='Tags')
final_dummy=dummy.drop(['Tags_Unknown'], 1)
df=pd.concat([df,final_dummy], axis=1)

df.shape
df=df.drop(['Lead Quality','Asymmetrique Activity Score','Asymmetrique Profile Score', 'Last Activity',
        'What is your current occupation', 'Lead Profile', 'Specialization',
        'City', 'Tags'], axis=1)

df.info()

numerical = df[['TotalVisits','Total Time Spent on Website',
                'Page Views Per Visit']]

numerical.describe()

# Plotting the Boxplots 
plt.figure(figsize = (20,10))
plt.subplot(2,2,1)
sns.boxplot(numerical['TotalVisits'])
plt.subplot(2,2,2)
sns.boxplot(numerical['Total Time Spent on Website'])
plt.subplot(2,2,3)
sns.boxplot(numerical['Page Views Per Visit'])

# Removing Outliers using IQR
df.describe()

Q1=df['TotalVisits'].quantile(0.25)
Q3=df['TotalVisits'].quantile(0.75)
IQR=Q3-Q1
df=df.loc[(df['TotalVisits'] >= Q1 - 1.5*IQR) & (df['TotalVisits'] <= Q3 + 1.5*IQR)]

Q1=df['Total Time Spent on Website'].quantile(0.25)
Q3=df['Total Time Spent on Website'].quantile(0.75)
IQR=Q3-Q1
df=df.loc[(df['Total Time Spent on Website']>=Q1-1.5*IQR) & (df['Total Time Spent on Website']<=Q3+1.5*IQR)]

Q1=df['Page Views Per Visit'].quantile(0.25)
Q3=df['Page Views Per Visit'].quantile(0.75)
IQR=Q3-Q1
df=df.loc[(df['Page Views Per Visit']>=Q1-1.5*IQR) & (df['Page Views Per Visit']<=Q3+1.5*IQR)]

# Visdualization after handling the outliers
plt.figure(figsize = (20,10))
plt.subplot(2,2,1)
sns.boxplot(df['TotalVisits'])
plt.subplot(2,2,2)
sns.boxplot(df['Total Time Spent on Website'])
plt.subplot(2,2,3)
sns.boxplot(df['Page Views Per Visit'])

df.head()
df.info()

# Scalling the Columns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler(0)

# Assigning X and Y as Target for Predictions
X=df.drop(['Lead Number', 'Converted'], axis=1)
X
y=df['Converted']
y

# Assigning and splitting the X and Y 
X_train, X_test, y_train, y_test=train_test_split(X,y, train_size=0.7, random_state=100)
X_train[['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit'
         ]]=scaler.fit_transform(X_train[['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']])

X_train.head()
y_train.head()
y.head()


# Using Stats Models to GLM summary
import statsmodels.api as sm

m1=sm.GLM(y_train, (sm.add_constant(X_train)),family=sm.families.Binomial())
m1.fit().summary()


# Applying Logistic Refression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()

# applying Recursive Feature Elimination -- RFE
from sklearn.feature_selection import RFE
rfe=RFE(lr,20)
rfe=rfe.fit(X_train, y_train)

rfe.support_

col=X_train.columns[rfe.support_]
col

X_train_sm=sm.add_constant(X_train[col])
m2=sm.GLM(y_train, X_train_sm,family=sm.families.Binomial())
reg=m2.fit()
reg.summary()

# Getting Predicted Value on train datasets
y_train_pred=reg.predict(X_train_sm)
y_train_pred[:15]

y_train_pred=y_train_pred.values.reshape(-1)
y_train_pred[:15]

y_train_pred_final=pd.DataFrame({'Converted':y_train.values, 'Conversion_prob': 
                                 y_train_pred})
y_train_pred_final['Lead ID']=y_train.index
y_train_pred_final.head()

y_train_pred_final['predisted']=y_train_pred_final.Conversion_prob.map(lambda x:1 if x>0.5 else 0)
y_train_pred_final.head()

# Confusion Metrix
from sklearn import metrics
confusion=metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predisted)
print(confusion)

# overall accuracy
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predisted))

# Checking VIF (Variation inflation factors)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables
# and their respective VIFs

vif=pd.DataFrame()
vif['Features']= X_train[col].columns
vif

vif['VIF']= [variance_inflation_factor(X_train[col].values,i) for i in range(
    X_train[col].shape[1])]

vif['VIF']=round(vif['VIF'],2)
vif=vif.sort_values(by='VIF', ascending=True)
vif

# now compare with GLM and VIF and start to drop columns having high VIF P-values

X_train_sm=sm.add_constant(X_train[col])
m2=sm.GLM(y_train, X_train_sm,family=sm.families.Binomial())
reg=m2.fit()
reg.summary()

col=col.drop(['Tags_wrong number given','Tags_number not provided', 'Tags_invalid number', 
              'Tags_Not doing further education'],1)


X_train_sm=sm.add_constant(X_train[col])
m2=sm.GLM(y_train, X_train_sm,family=sm.families.Binomial())
reg=m2.fit()
reg.summary()


confusion=metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predisted)
print(confusion)

print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predisted))
# Final Accuracy is 92% appx

# Correlation Metrix
plt.figure(figsize = (20,10),dpi=200)
sns.heatmap(X_train[col].corr(), annot=True)


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
'''
# Calculation of Spacificity - TN/TN+FP
Sensitivity = TP/TP+FN
Positive Predictive Value = TP/TP+FP
Negative Predicted Value = TN/TN+FN
'''

Spacificity = round(TN/(TN+FP),2)
Sensitivity= round(TP/(TP+FN),2)
Positive_Predictive=round(TP/(TP+FP),2)
Negative_Predictive=round(TN/(TN+FN),2)

print('Specificity Accuracy is : {}'.format(Spacificity))
print('Sensitivity Accuracy is :{}'.format(Sensitivity))
print('Positive_Predictive is : {}'.format(Positive_Predictive))
print('Negative_Predictive is : {}'.format(Negative_Predictive))

