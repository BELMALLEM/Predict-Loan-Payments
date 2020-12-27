
<h1 align="center"><font size="5">Loan Payments Prediction</font></h1>

In this notebook we'll try to presdict loan payments for given clients.

We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.

Lets first load required libraries:


```python
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
%matplotlib inline
```

### About dataset

This dataset is about past loans. The __Loan_train.csv__ data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:

| Field          | Description                                                                           |
|----------------|---------------------------------------------------------------------------------------|
| Loan_status    | Whether a loan is paid off or in collection                                           |
| Principal      | Basic principal loan amount at the                                                    |
| Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
| Effective_date | When the loan got originated and took effects                                         |
| Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
| Age            | Age of applicant                                                                      |
| Education      | Education of applicant                                                                |
| Gender         | The gender of applicant                                                               |

Lets download the dataset


```python
!wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv
```

### Load Data From CSV File  


```python
df = pd.read_csv('loan_train.csv')
df.head()
```


```python
df.shape
```

### Convert to date time object 


```python
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()
```

# Data visualization and pre-processing



Let’s see how many of each class is in our data set 


```python
df['loan_status'].value_counts()
```

260 people have paid off the loan on time while 86 have gone into collection 


Lets plot some columns to underestand data better:


```python
# notice: installing seaborn might takes a few minutes
!conda install -c anaconda seaborn -y
```


```python
import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()
```


```python
bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()
```

# Pre-processing:  Feature selection/extraction

### Lets look at the day of the week people get the loan 


```python
df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

```

We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less then day 4 


```python
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()
```

## Convert Categorical features to numerical values

Lets look at gender:


```python
df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
```

86 % of female pay there loans while only 73 % of males pay there loan


Lets convert male to 0 and female to 1:



```python
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()
```

## One Hot Encoding  
#### How about education?


```python
df.groupby(['education'])['loan_status'].value_counts(normalize=True)
```

#### Feature befor One Hot Encoding


```python
df[['Principal','terms','age','Gender','education']].head()
```

#### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame 


```python
Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature, pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1, inplace=True)
Feature.head()

```

### Feature selection

Lets defind feature sets, X:


```python
X = Feature
X[0:5]
```

What are our lables?


```python
y = df['loan_status'].values
y[0:5]
```

## Normalize Data 

Data Standardization give data zero mean and unit variance (technically should be done after train test split )


```python
X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]
```

# Classification 

Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model
You should use the following algorithm:
- K Nearest Neighbor(KNN)
- Decision Tree
- Support Vector Machine
- Logistic Regression



__ Notice:__ 
- You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
- You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
- You should include the code of the algorithm in the following cells.


```python
from sklearn import metrics
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
```

# K Nearest Neighbor(KNN)
Notice: You should find the best k to build the model with the best accuracy.  
**warning:** You should not use the __loan_test.csv__ for finding the best k, however, you can split your train_loan.csv into train and test to find the best __k__.


```python
from sklearn.neighbors import KNeighborsClassifier
```

## Training:


```python
Ks = 10
mean_acc = np.zeros(Ks-1)
std_acc = np.zeros(Ks-1)
ConfusionMX = []
for n in range(1, Ks):
    # We'll train the model and predict the result
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train, y_train)
    y_predict = neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, y_predict)
    std_acc[n-1] = np.std(y_predict == y_test)/np.sqrt(y_predict.shape[0])
    
```


```python
plt.plot(range(1, Ks), mean_acc, 'g')
plt.fill_between(range(1,Ks), mean_acc - 1*std_acc, mean_acc + 1*std_acc, alpha=0.10)
plt.legend('accuracy', '+/- 3xstd')
plt.ylabel('accuracy')
plt.xlabel('number of neighbors (K)')
plt.tight_layout()
plt.show()
```


```python
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1 )
```

So the best k for our KNN model is k=7


```python
k = 7
KNNeighbors = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
```

# Decision Tree


```python
from sklearn.tree import DecisionTreeClassifier

treeModel = DecisionTreeClassifier()
treeModel.fit(X_train, y_train)
```

# Support Vector Machine


```python
from sklearn import svm
```


```python
SVM = svm.SVC(kernel='rbf').fit(X_train, y_train)
```

# Logistic Regression


```python
from sklearn.linear_model import LogisticRegression
```


```python
LogistReg = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
```

# Model Evaluation using Test set


```python
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
```

First, download and load the test set:

!wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv

### Load Test set for evaluation 


```python
test_df = pd.read_csv('loan_test.csv')
test_df.head()
```


```python
test_df['due_date'] = pd.to_datetime(df['due_date'])
test_df['effective_date'] = pd.to_datetime(df['effective_date'])
```


```python
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
```


```python
Feature = test_df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature, pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1, inplace=True)
Feature.head()
```


```python
X_test_set = Feature[['Principal','terms','age','Gender','weekend']]
X_test_set = preprocessing.StandardScaler().fit(X).transform(X)
X_test_set = X_test_set[:53]

y_test_set = test_df['loan_status'].values
y_test_set = y_test_set[:53]
```

### Evaluating the KNN model


```python
predictionKNN = KNNeighbors.predict(X_test_set)
```


```python
print('K-Neireast-Neighbors Model Evaluation: ')
print('Jaccard Score: ', jaccard_similarity_score(y_test_set, predictionKNN))
print('F1_Score: ', f1_score(y_test_set, predictionKNN, average='weighted'))
```

### Evaluating the SVM model


```python
predictionSVM = SVM.predict(X_test_set)
```


```python
print('Support Vectore Machine Model Evaluation: ')
print('Jaccard Score: ', jaccard_similarity_score(y_test_set, predictionSVM))
print('F1_Score: ', f1_score(y_test_set, predictionSVM, average='weighted'))
```

### Evaluating the DecisionTree Model


```python
predictionTree = treeModel.predict(X_test_set)
```


```python
print('Decision Tree Model Evaluation: ')
print('Jaccard Score: ', jaccard_similarity_score(y_test_set, predictionTree))
print('F1_Score: ', f1_score(y_test_set, predictionTree, average='weighted'))
```

### Evaluating the Logistic Regression Model


```python
predictionLogisticReg = LogistReg.predict(X_test_set)
predictionLogisticReg_proba = LogistReg.predict_proba(X_test_set)
```


```python
print('Logistic Regression Model Evaluation: ')
print('Jaccard Score: ', jaccard_similarity_score(y_test_set, predictionLogisticReg))
print('F1_Score: ', f1_score(y_test_set, predictionLogisticReg, average='weighted'))
```

### Change string values with numeric ones for LogLoss evaluation

New sets one for test and another for the predicted one


```python
y_test_LogLoss = test_df['loan_status'][:53]
prediction_LogLoss = predictionLogisticReg_proba
```

change the values


```python
y_test_LogLoss = np.asarray(y_test_LogLoss)
y_test_LogLoss[ y_test_LogLoss == 'PAIDOFF' ] = 0
y_test_LogLoss[ y_test_LogLoss == 'COLLECTION' ] = 1
y_test_LogLoss = y_test_LogLoss.astype('int')
```


```python
log_loss(y_test_LogLoss, prediction_LogLoss)
```

# Report
You should be able to report the accuracy of the built model using different evaluation metrics:

| Algorithm          | Jaccard                 | F1-score                  | LogLoss                  |
|--------------------|-------------------------|---------------------------|--------------------------|
| KNN                | 0.5660377358490566      | 0.5713274774655239        | NA                       |
| Decision Tree      | 0.6415094339622641      | 0.6458792205149981        | NA                       |
| SVM                | 0.6981132075471698      | 0.6892161374443933        | NA                       |
| LogisticRegression | 0.6415094339622641      | 0.6365525273701375        | 0.7483864618819982       |

## Thanks for reading this notebook  :))
