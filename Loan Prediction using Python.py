#!/usr/bin/env python
# coding: utf-8

# Step 1.Problem Statement
# 
# Understand the problem statement is the first and foremost step.This would help you give an intuition of what you will face ahead of time. Lets us see the problem statement
# 
# Dream Housing Finance Company deals in all home loans.They have presence all urban,semi urban and rural areas.Customer first apply for home loan after that company validates the customer eligibility for loan.Company want to automate the loan eligibility process(real time)based on custer detail provided while filling online application form.These details are Gender,Marital status,Education,Number of Dependent,Income,Loan Amount,Credit,History and other.
# To automate this process,they have given a problem to identify the customer segments,those are eligible for loan amount so that they can specifically target these customers.
# 
# This is classification problem where we have to predict whether a loan would be approved or not.

# Step2: Hypothesis generation
# 
# After looking at the problem statement,we will now move into hypothesis generation.it is the process of listing out all the possible factor that can affect the outcome.
# 
# Below are some of the factor which can affect Loan Approval.
# 
# 1.Salary: Appicant with high Income should have more chances of loan approval.
# 
# 2.Previous history:Applicants who have repayed their previous debt should have higher chance of loan approval.
# 
#     
# 3.Loan Amount:Loan approval should also dependent on loan amount.if the loan amount is less, the chance of loan approval should be high.
# 
# 4.Loan terms: loan for less amount and less less term have high chances of approval.
# 
# 5.EMI:lesser the amount to be paid monthly to repay the loan,higher the chance of loan approval.

# Step 3:Loading the data
#     

# In[2]:


import numpy as np                     #for mathematical calculations
import pandas as pd                    #for data analysis
import matplotlib.pyplot as plt         #for plotting graphs
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns                  #for data visualization

import warnings                         #To ignore any warnings
warnings.filterwarnings("ignore")


# In[4]:


train = pd.read_csv("train_l.csv")
test = pd.read_csv("test_l.csv")


# In[5]:


#Let's make a copy of data so that even if we have to make any changes in these datasets we would not lose the original datasets.

train_original = train.copy()
test_original = test.copy()


# Step 3: Data Preprocessing

# In[7]:


#Understand the data set

train.head()


# In[8]:


train.columns


# In[9]:


test.columns


# In[10]:


train.describe()


# In[12]:


train.info()


# In[14]:


train.shape


# In[15]:


test.shape


# Univariate Analysis
# 
# For Target Variable

# In[18]:


train['Loan_Status'].value_counts()


# In[19]:


#Normalize can be set to True to print proportions instead of number 

train['Loan_Status'].value_counts(normalize=True)


# In[21]:


#Visualize the Target variable

train['Loan_Status'].value_counts().plot.bar()


# The loan of 422(69%) of people out of 614 was approved

# #Now let's visualize each variable separately .Different types of variable are categorical, ordinal and nominal
# * Categorical Feature: These feature have categories (Gender,Married,Self_Employed,Credit_History,Loan_Status)
# *Ordinal Feature:Variable in categorical features having some order involved(Dependents,Education,Property_Area)
# *Numerical Features:These features have numerical values(ApplicantIncome,CoapplicantIncome,LoanAmount)
#     

# In[32]:


#Indipendent Variable(Categorical)

plt.figure(1)
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20, 15), title='Gender')

plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(title='Married')

plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(title='Self_Employed')

plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(title='Credit_History')


# It can inferrd from above bar plots that:
#     
# *80% applicant in dataset are male
# 
# *Around 65% of applicants in the dataset are married
# 
# *Around 15% of applicants in the dataset are self employed
# 
# *Around 85% of applicants have repaid their debts.
# 
# 

# # Now lets visualize the ordinal variable:

# Independent Variable(Ordinal)

# In[34]:


plt.figure(1)
plt.subplot(131)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6),title='Dependents')

plt.subplot(132)
train['Education'].value_counts(normalize=True).plot.bar(title='Education')

plt.subplot(133)
train['Property_Area'].value_counts(normalize=True).plot.bar(title='Property_Area')
plt.show()


# Following inferances can be made from the above bar plots:
# 
# *Most of the applicants don't have any dependents.
# 
# *Around 80%of the applicants are graduate.
# 
# *Most of the applicantsa are from semiurban area.

# # Indepent Variable(Numerical)

# In[36]:


plt.figure(1)
plt.subplot(121)
sns.distplot(train['ApplicantIncome']);
plt.subplot(122) 
train['ApplicantIncome'].plot.box(figsize=(16,5))
plt.show()


# In[39]:


#Let's segrate them by education

train.boxplot(column='ApplicantIncome',by='Education')
plt.suptitle("")


# In[45]:


#Let's Look coapplicant income distribution

plt.figure(1)
plt.subplot(121)
sns.distplot(train['CoapplicantIncome']);

plt.subplot(122)
train['CoapplicantIncome'].plot.box(figsize=(16,5))
plt.show()


# In[46]:


#Let's Look at the LoanAmount Variable

plt.figure(1)
plt.subplot(121)
df=train.dropna()
sns.distplot(train['LoanAmount']);

plt.subplot(122)
train['LoanAmount'].plot.box(figsize=(16,5))
plt.show()


# We see lot of outlier in this variable and the distribution is fairly normal.we will trat the outliers in later sections

# # Bivariate Analysis

# Let's recall some of the hypothesis that we generate earlier:
# 
# 1.Salary: Appicant with high Income should have more chances of loan approval.
# 
# 2.Previous history:Applicants who have repayed their previous debt should have higher chance of loan approval.
# 
# 3.Loan Amount:Loan approval should also dependent on loan amount.if the loan amount is less, the chance of loan approval should be high.
# 
# 4.Loan terms: loan for less amount and less less term have high chances of approval.
# 
# 5.EMI:lesser the amount to be paid monthly to repay the loan,higher the chance of loan approval.

# Let's try to check the above mentioned hypotheses using bivariate analysis
# 
# After looking at every variable indivisually in univariate analysis, we will now explore them again with respoect to target variable.

# # Categorical Independent Variable vs Target Variable

# In[49]:


Gender = pd.crosstab(train['Gender'],train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True,figsize=(4,4))


# In[50]:


#Now let's visualize the remaining categorical variables vs target variable.

Married = pd.crosstab(train['Married'],train['Loan_Status'])
Dependents=pd.crosstab(train['Dependents'],train['Loan_Status'])
Education=pd.crosstab(train['Education'],train['Loan_Status'])
Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status'])

Married.div(Married.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True,figsize=(4,4))
plt.show()
Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True,figsize=(4,4))
plt.show()
Education.div(Education.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True,figsize=(4,4))
plt.show()
Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True,figsize=(4,4))
plt.show()


# #Now we will look at the relationship between remaining categorical independent variables and Loan_Status.

# In[55]:


Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status'])
Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status'])

Credit_History.div(Credit_History.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True,figsize=(4,4))
plt.show()
Property_Area.div(Property_Area.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True,figsize=(4,4))
plt.show()


# Now Let's Visualize Numerical Independent with respect to target variable
# 
# Numerical Independent Variable Vs Target Variable

# #We will try to find out mean Income of people for which the Loan has been approved vs the mean income of people of people for which the has not been approved

# In[59]:


train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()


# Here the Y-axis repersent the meean Income of Applicant.we don't see any change in the mean income.So let us make bins
# for the applicant income variable based on the values in it and analyze the crossponding loan status for each bin.

# In[61]:


bins = [0,2500,4000,6000,81000]
group=['Low', 'Average', 'High', 'Very High']
train['Income_bin']=pd.cut(train['ApplicantIncome'],bins,labels=group)

Income_bin = pd.crosstab(train['Income_bin'],train['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
plt.xlabel("ApplicantINcome")
plt.ylabel("Percentage")


# It can inferred that Applicant Income does not affect the chances of loan approval which contradict our hypothesis in which
# we assume that if the applicant income is high the chances of loan approval will also high

# We will analyse the coapplicant income and Loan amount variable in similar Manner.

# In[66]:


bins = [0,1000,3000,42000]
group = ['Low','Average','High']

train['Coapplicant_Income_bin']=pd.cut(train['CoapplicantIncome'],bins,labels = group)
Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'])

Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
plt.xlabel=('CoapplicantIncome')
plt.ylabel=('Percentage')


# It shows the coapplicant income is low the chances of loan approval are high.But this does not look right.The Possible reason
# behind this may be most of the applicant does not have any coappliacnt so that the coapplicant income for such applicant are 0 and 
# hence loan approval are not dependent on it.So we can make a new variable in which we will combine the applicants and coapplcants
# income to visualize the combined effect of income on loan approval 

# #Let's combine the Applicant Income and Coapplicant Income and see the combined effect of Total Income on the Loan_Status

# In[72]:


train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
bins=[0,2500,4000,6000,81000]
group = ['Low', 'Average', 'High', 'Very High']

train['Total_Income_bin'] = pd.cut(train['Total_Income'],bins,labels=group)

Total_Income_bin=pd.crosstab(train['Total_Income_bin'],train['Loan_Status'])
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
plt.xlabel=('Total_Income')
plt.ylabel=('Percentage')


# As we can see that Proportion of loans getting approved for applicants having low,Total_Income is very less as compared
# to the applicant with average ,High and Very High Income.

# Now Let's see Loan Amount Variables

# In[76]:


bins = [0,100,200,700]
group = ['Low', 'Average', 'High']
train['LoanAmount_bin'] = pd.cut(train['LoanAmount'], bins, labels=group)

LoanAmount_bin = pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'])
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
plt.xlabel=('Loan_Amount')
plt.ylabel = ("Percentage")


# It can be seen that the proportion of approval loans is higher for Low and Average Loan Amount as compared to that of 
# High Loan Amount which supports our hypothesis in which we considered that the chances of loan approval will be when the 
# loan amount will be low.

# Now Let's Drop the the bins which we created for the exploration part. we will change the 3+ in dependents variable to
# 3 to make it a numerical variable.we will also convert the target variable's categories into 0 and 1.so that we can find
# its corelation with numerical variables.
# 
# One more reason to do so is few model like logistic regression takes only numeric values as input.we will replace N with 0
# and Y with 1

# In[77]:


train = train.drop(['Income_bin',"Coapplicant_Income_bin","LoanAmount_bin", "Total_Income_bin","Total_Income"],axis=1)


# In[79]:


train['Dependents'].replace('3+' , 3,inplace = True)
test['Dependents'].replace('3+' , 3,inplace = True)
train['Loan_Status'].replace('N' , 0,inplace = True)
train['Loan_Status'].replace('Y', 1,inplace=True)


# Now lets look at the correlation between all the numerical veriables.we will use the heat map to visualize the correlation
# 
# Heatmap visualize the data through variation in coloring. The variable with darker color means their correlation is more

# In[81]:


matrix = train.corr()
f, ax = plt.subplots(figsize = (9,6))
sns.heatmap(matrix, vmax = 0.8, square = True, cmap = "BuPu");


# # Step 4: Missing value and outlier treatment

# Missing Value Imputation

# In[82]:


train.isnull().sum()


# We will treat the missing values in all the feature one by one

# We can consider these methods to fill the missing values:
# 
#     * For Numerical Variable : Imputation using mean or median
#     
#     * For categorical Variable : Imputation using mode

# In[88]:


train["Gender"].fillna(train['Gender'].mode()[0],inplace = True)
train["Married"].fillna(train['Married'].mode()[0],inplace = True)
train["Dependents"].fillna(train['Dependents'].mode()[0],inplace = True)
train["Self_Employed"].fillna(train['Self_Employed'].mode()[0],inplace = True)
train["Credit_History"].fillna(train['Credit_History'].mode()[0],inplace = True)


# In[89]:


train['Loan_Amount_Term'].value_counts()


# In[90]:


train["Loan_Amount_Term"].fillna(train['Loan_Amount_Term'].mode()[0],inplace = True)


# we saw loan amount have outliers ,so we will use median value to treat the outliers beacause mean is not good way to treat this

# In[91]:


train['LoanAmount'].fillna(train['LoanAmount'].median(),inplace = True)


# In[92]:


train.isnull().sum()


# In[93]:


test.isnull().sum()


# As we see all missing values are trated in train dataset
# 
# Now Let's fill the missing value in test dataset with same approach

# In[94]:


test["Gender"].fillna(test['Gender'].mode()[0],inplace = True)
test["Married"].fillna(test['Married'].mode()[0],inplace = True)
test["Dependents"].fillna(test['Dependents'].mode()[0],inplace = True)
test["Self_Employed"].fillna(test['Self_Employed'].mode()[0],inplace = True)
test["Credit_History"].fillna(test['Credit_History'].mode()[0],inplace = True)
test["Loan_Amount_Term"].fillna(test['Loan_Amount_Term'].mode()[0],inplace = True)
test['LoanAmount'].fillna(test['LoanAmount'].median(),inplace = True)


# In[95]:


test.isnull().sum()


# #Outlier Treatment

# Due to the bulk of the data in LoanAmount is at the left and the right tail is longer.This is called right skewness.
# 
# One way to remove the skewness is by doing the log transformation. As we take the log transformation .it does not affect
# 
# the smaller values much,but reduce the larger value. So we get a distribution similer to normal distribution

# In[96]:


train['LoanAmount_log'] = np.log(train['LoanAmount'])


# In[97]:


train['LoanAmount_log'].hist(bins=20)


# In[98]:


#same for test data

test["LoanAmount_log"] = np.log(test['LoanAmount'])


# Now distribution looks much closer to normal and effect of extreme values has been significantly subsided.

# # Model Building Part 1

# In[103]:


#Let's drop Loan_Id Variable as it do not have any effect on the loan status

train = train.drop("Loan_ID", axis=1)
test = test.drop("Loan_ID", axis=1)


# Now we Sklearn require the target variable in a separate dataset. So we will drop out target variable from train dataset and save it into another dataset

# In[104]:


X = train.drop("Loan_Status" , axis = 1)
y = train.Loan_Status


# In[105]:


X.columns


# Now we will make dummy Variables for the categorical variables.Dummy Variable turns categorical variables into a series
# 
# of 0 and 1,making them easier to quantify and compare.
# 
# As a logistics regression takes numerical values as input

# In[111]:


X = pd.get_dummies(X)
train = pd.get_dummies(train)
test=pd.get_dummies(test)


# Now let us divide the dataset into train and validation set

# In[112]:


from sklearn.model_selection import train_test_split

X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size = 0.3)


# # Logistic Regression

# In[114]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)


# In[119]:


pred_cv =model.predict(X_cv)


# In[120]:


from sklearn.metrics import accuracy_score
accuracy_score(y_cv,pred_cv)


# so our prediction is almost 78% accurate,we have identified 78% loan status correctly

# In[122]:


#Now let's prediction for test dataset

y_pred = model.predict(test)


# In[128]:


#Now lets evaluate the model performance

from sklearn.metrics import confusion_matrix

#Let's print the confudion matrix first
plt.rcParams['figure.figsize'] = (10 , 10)
cm = confusion_matrix(y_cv, pred_cv)
sns.heatmap(cm ,annot = True , cmap = 'Wistia')
plt.title("Confusion Metrix for Logistic Regression",fontsize = 15)
plt.show()


# In[126]:


y.shape


# In[127]:


y_pred.shape


# In[130]:


#Let's print the classification report also
from sklearn.metrics import classification_report
cr = classification_report(pred_cv, y_cv)
print(cr)


# In[131]:


train.head()


# In[ ]:




