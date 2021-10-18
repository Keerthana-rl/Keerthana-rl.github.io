---
title:  "Churn Prediction "
permalink: /posts/logisticregression/
excerpt: "Week-3 covers churn prediction"
last_modified_at: 2021-09-19T16:00:11-04:00
header:
  #image: assets/images/week-2/20210919-car.jpg
  teaser: assets/images/week-2/20210919-car.jpg
categories:
- tutorial
tags:
- mlzoomcamp
toc: true
toc_sticky: true
#classes: wide
---


# 3. Machine Learning for Classification

We'll use logistic regression to predict churn


## 3.1 Churn prediction project

* Dataset: https://www.kaggle.com/blastchar/telco-customer-churn
* https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv


## 3.2 Data preparation

* Download the data, read it with pandas
* Look at the data
* Make column names and values look uniform
* Check if all the columns read correctly
* Check if the churn variable needs any preparation


```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
```


```python
data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'
```


```python
df = pd.read_csv(r'E:\gito\mlbookcamp-code\chapter-03-churn-prediction\WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customerID</th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>...</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7590-VHVEG</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>1</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5575-GNVDE</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>34</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>56.95</td>
      <td>1889.5</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3668-QPYBK</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7795-CFOCW</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>45</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1840.75</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9237-HQITU</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')
```


```python
df.head().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>customerid</th>
      <td>7590-vhveg</td>
      <td>5575-gnvde</td>
      <td>3668-qpybk</td>
      <td>7795-cfocw</td>
      <td>9237-hqitu</td>
    </tr>
    <tr>
      <th>gender</th>
      <td>female</td>
      <td>male</td>
      <td>male</td>
      <td>male</td>
      <td>female</td>
    </tr>
    <tr>
      <th>seniorcitizen</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>partner</th>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
    </tr>
    <tr>
      <th>dependents</th>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
    </tr>
    <tr>
      <th>tenure</th>
      <td>1</td>
      <td>34</td>
      <td>2</td>
      <td>45</td>
      <td>2</td>
    </tr>
    <tr>
      <th>phoneservice</th>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>no</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>multiplelines</th>
      <td>no_phone_service</td>
      <td>no</td>
      <td>no</td>
      <td>no_phone_service</td>
      <td>no</td>
    </tr>
    <tr>
      <th>internetservice</th>
      <td>dsl</td>
      <td>dsl</td>
      <td>dsl</td>
      <td>dsl</td>
      <td>fiber_optic</td>
    </tr>
    <tr>
      <th>onlinesecurity</th>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>yes</td>
      <td>no</td>
    </tr>
    <tr>
      <th>onlinebackup</th>
      <td>yes</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
    </tr>
    <tr>
      <th>deviceprotection</th>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
    </tr>
    <tr>
      <th>techsupport</th>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
    </tr>
    <tr>
      <th>streamingtv</th>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
    </tr>
    <tr>
      <th>streamingmovies</th>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
    </tr>
    <tr>
      <th>contract</th>
      <td>month-to-month</td>
      <td>one_year</td>
      <td>month-to-month</td>
      <td>one_year</td>
      <td>month-to-month</td>
    </tr>
    <tr>
      <th>paperlessbilling</th>
      <td>yes</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>paymentmethod</th>
      <td>electronic_check</td>
      <td>mailed_check</td>
      <td>mailed_check</td>
      <td>bank_transfer_(automatic)</td>
      <td>electronic_check</td>
    </tr>
    <tr>
      <th>monthlycharges</th>
      <td>29.85</td>
      <td>56.95</td>
      <td>53.85</td>
      <td>42.3</td>
      <td>70.7</td>
    </tr>
    <tr>
      <th>totalcharges</th>
      <td>29.85</td>
      <td>1889.5</td>
      <td>108.15</td>
      <td>1840.75</td>
      <td>151.65</td>
    </tr>
    <tr>
      <th>churn</th>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
tc = pd.to_numeric(df.totalcharges, errors='coerce')
```


```python
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
```


```python
df.totalcharges = df.totalcharges.fillna(0)
```


```python
df.churn.head()
```




    0     no
    1     no
    2    yes
    3     no
    4    yes
    Name: churn, dtype: object




```python
df.churn = (df.churn == 'yes').astype(int)
```

## 3.3 Setting up the validation framework

* Perform the train/validation/test split with Scikit-Learn


```python
from sklearn.model_selection import train_test_split
```


```python
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
```


```python
len(df_train), len(df_val), len(df_test)
```




    (4225, 1409, 1409)




```python
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
```


```python
y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']
```

## 3.4 EDA

* Check missing values
* Look at the target variable (churn)
* Look at numerical and categorical variables


```python
df_full_train = df_full_train.reset_index(drop=True)
```


```python
df_full_train.isnull().sum()
```




    customerid          0
    gender              0
    seniorcitizen       0
    partner             0
    dependents          0
    tenure              0
    phoneservice        0
    multiplelines       0
    internetservice     0
    onlinesecurity      0
    onlinebackup        0
    deviceprotection    0
    techsupport         0
    streamingtv         0
    streamingmovies     0
    contract            0
    paperlessbilling    0
    paymentmethod       0
    monthlycharges      0
    totalcharges        0
    churn               0
    dtype: int64




```python
df_full_train.churn.value_counts(normalize=True)
```




    0    0.730032
    1    0.269968
    Name: churn, dtype: float64




```python
df_full_train.churn.mean()
```




    0.26996805111821087




```python
numerical = ['tenure', 'monthlycharges', 'totalcharges']
```


```python
categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]
```


```python
df_full_train[categorical].nunique()
```




    gender              2
    seniorcitizen       2
    partner             2
    dependents          2
    phoneservice        2
    multiplelines       3
    internetservice     3
    onlinesecurity      3
    onlinebackup        3
    deviceprotection    3
    techsupport         3
    streamingtv         3
    streamingmovies     3
    contract            3
    paperlessbilling    2
    paymentmethod       4
    dtype: int64



## 3.5 Feature importance: Churn rate and risk ratio

Feature importance analysis (part of EDA) - identifying which features affect our target variable

* Churn rate
* Risk ratio
* Mutual information - later

<img src="assets/images/churn risk.png" width=400 height=400 />

#### Churn rate


```python
df_full_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customerid</th>
      <th>gender</th>
      <th>seniorcitizen</th>
      <th>partner</th>
      <th>dependents</th>
      <th>tenure</th>
      <th>phoneservice</th>
      <th>multiplelines</th>
      <th>internetservice</th>
      <th>onlinesecurity</th>
      <th>...</th>
      <th>deviceprotection</th>
      <th>techsupport</th>
      <th>streamingtv</th>
      <th>streamingmovies</th>
      <th>contract</th>
      <th>paperlessbilling</th>
      <th>paymentmethod</th>
      <th>monthlycharges</th>
      <th>totalcharges</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5442-pptjy</td>
      <td>male</td>
      <td>0</td>
      <td>yes</td>
      <td>yes</td>
      <td>12</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>no_internet_service</td>
      <td>...</td>
      <td>no_internet_service</td>
      <td>no_internet_service</td>
      <td>no_internet_service</td>
      <td>no_internet_service</td>
      <td>two_year</td>
      <td>no</td>
      <td>mailed_check</td>
      <td>19.70</td>
      <td>258.35</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6261-rcvns</td>
      <td>female</td>
      <td>0</td>
      <td>no</td>
      <td>no</td>
      <td>42</td>
      <td>yes</td>
      <td>no</td>
      <td>dsl</td>
      <td>yes</td>
      <td>...</td>
      <td>yes</td>
      <td>yes</td>
      <td>no</td>
      <td>yes</td>
      <td>one_year</td>
      <td>no</td>
      <td>credit_card_(automatic)</td>
      <td>73.90</td>
      <td>3160.55</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2176-osjuv</td>
      <td>male</td>
      <td>0</td>
      <td>yes</td>
      <td>no</td>
      <td>71</td>
      <td>yes</td>
      <td>yes</td>
      <td>dsl</td>
      <td>yes</td>
      <td>...</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>two_year</td>
      <td>no</td>
      <td>bank_transfer_(automatic)</td>
      <td>65.15</td>
      <td>4681.75</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6161-erdgd</td>
      <td>male</td>
      <td>0</td>
      <td>yes</td>
      <td>yes</td>
      <td>71</td>
      <td>yes</td>
      <td>yes</td>
      <td>dsl</td>
      <td>yes</td>
      <td>...</td>
      <td>yes</td>
      <td>yes</td>
      <td>yes</td>
      <td>yes</td>
      <td>one_year</td>
      <td>no</td>
      <td>electronic_check</td>
      <td>85.45</td>
      <td>6300.85</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2364-ufrom</td>
      <td>male</td>
      <td>0</td>
      <td>no</td>
      <td>no</td>
      <td>30</td>
      <td>yes</td>
      <td>no</td>
      <td>dsl</td>
      <td>yes</td>
      <td>...</td>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>no</td>
      <td>one_year</td>
      <td>no</td>
      <td>electronic_check</td>
      <td>70.40</td>
      <td>2044.75</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
churn_female = df_full_train[df_full_train.gender == 'female'].churn.mean()
churn_female
```




    0.27682403433476394




```python
churn_male = df_full_train[df_full_train.gender == 'male'].churn.mean()
churn_male
```




    0.2632135306553911




```python
global_churn = df_full_train.churn.mean()
global_churn
```




    0.26996805111821087




```python
global_churn - churn_female
```




    -0.006855983216553063




```python
global_churn - churn_male
```




    0.006754520462819769




```python
df_full_train.partner.value_counts()
```




    no     2932
    yes    2702
    Name: partner, dtype: int64




```python
churn_partner = df_full_train[df_full_train.partner == 'yes'].churn.mean()
churn_partner
```




    0.20503330866025166




```python
global_churn - churn_partner
```




    0.06493474245795922




```python
churn_no_partner = df_full_train[df_full_train.partner == 'no'].churn.mean()
churn_no_partner
```




    0.3298090040927694




```python
global_churn - churn_no_partner
```




    -0.05984095297455855



#### Risk ratio


```python
churn_no_partner / global_churn
```




    1.2216593879412643




```python
churn_partner / global_churn
```




    0.7594724924338315



```
SELECT
    gender,
    AVG(churn),
    AVG(churn) - global_churn AS diff,
    AVG(churn) / global_churn AS risk
FROM
    data
GROUP BY
    gender;
```


```python
from IPython.display import display
```


```python
for c in categorical:
    print(c)
    df_group = df_full_train.groupby(c).churn.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_churn
    df_group['risk'] = df_group['mean'] / global_churn
    display(df_group)
    print()
    print()
```

    gender
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>count</th>
      <th>diff</th>
      <th>risk</th>
    </tr>
    <tr>
      <th>gender</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>0.276824</td>
      <td>2796</td>
      <td>0.006856</td>
      <td>1.025396</td>
    </tr>
    <tr>
      <th>male</th>
      <td>0.263214</td>
      <td>2838</td>
      <td>-0.006755</td>
      <td>0.974980</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    seniorcitizen
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>count</th>
      <th>diff</th>
      <th>risk</th>
    </tr>
    <tr>
      <th>seniorcitizen</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.242270</td>
      <td>4722</td>
      <td>-0.027698</td>
      <td>0.897403</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.413377</td>
      <td>912</td>
      <td>0.143409</td>
      <td>1.531208</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    partner
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>count</th>
      <th>diff</th>
      <th>risk</th>
    </tr>
    <tr>
      <th>partner</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>no</th>
      <td>0.329809</td>
      <td>2932</td>
      <td>0.059841</td>
      <td>1.221659</td>
    </tr>
    <tr>
      <th>yes</th>
      <td>0.205033</td>
      <td>2702</td>
      <td>-0.064935</td>
      <td>0.759472</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    dependents
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>count</th>
      <th>diff</th>
      <th>risk</th>
    </tr>
    <tr>
      <th>dependents</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>no</th>
      <td>0.313760</td>
      <td>3968</td>
      <td>0.043792</td>
      <td>1.162212</td>
    </tr>
    <tr>
      <th>yes</th>
      <td>0.165666</td>
      <td>1666</td>
      <td>-0.104302</td>
      <td>0.613651</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    phoneservice
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>count</th>
      <th>diff</th>
      <th>risk</th>
    </tr>
    <tr>
      <th>phoneservice</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>no</th>
      <td>0.241316</td>
      <td>547</td>
      <td>-0.028652</td>
      <td>0.893870</td>
    </tr>
    <tr>
      <th>yes</th>
      <td>0.273049</td>
      <td>5087</td>
      <td>0.003081</td>
      <td>1.011412</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    multiplelines
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>count</th>
      <th>diff</th>
      <th>risk</th>
    </tr>
    <tr>
      <th>multiplelines</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>no</th>
      <td>0.257407</td>
      <td>2700</td>
      <td>-0.012561</td>
      <td>0.953474</td>
    </tr>
    <tr>
      <th>no_phone_service</th>
      <td>0.241316</td>
      <td>547</td>
      <td>-0.028652</td>
      <td>0.893870</td>
    </tr>
    <tr>
      <th>yes</th>
      <td>0.290742</td>
      <td>2387</td>
      <td>0.020773</td>
      <td>1.076948</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    internetservice
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>count</th>
      <th>diff</th>
      <th>risk</th>
    </tr>
    <tr>
      <th>internetservice</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>dsl</th>
      <td>0.192347</td>
      <td>1934</td>
      <td>-0.077621</td>
      <td>0.712482</td>
    </tr>
    <tr>
      <th>fiber_optic</th>
      <td>0.425171</td>
      <td>2479</td>
      <td>0.155203</td>
      <td>1.574895</td>
    </tr>
    <tr>
      <th>no</th>
      <td>0.077805</td>
      <td>1221</td>
      <td>-0.192163</td>
      <td>0.288201</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    onlinesecurity
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>count</th>
      <th>diff</th>
      <th>risk</th>
    </tr>
    <tr>
      <th>onlinesecurity</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>no</th>
      <td>0.420921</td>
      <td>2801</td>
      <td>0.150953</td>
      <td>1.559152</td>
    </tr>
    <tr>
      <th>no_internet_service</th>
      <td>0.077805</td>
      <td>1221</td>
      <td>-0.192163</td>
      <td>0.288201</td>
    </tr>
    <tr>
      <th>yes</th>
      <td>0.153226</td>
      <td>1612</td>
      <td>-0.116742</td>
      <td>0.567570</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    onlinebackup
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>count</th>
      <th>diff</th>
      <th>risk</th>
    </tr>
    <tr>
      <th>onlinebackup</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>no</th>
      <td>0.404323</td>
      <td>2498</td>
      <td>0.134355</td>
      <td>1.497672</td>
    </tr>
    <tr>
      <th>no_internet_service</th>
      <td>0.077805</td>
      <td>1221</td>
      <td>-0.192163</td>
      <td>0.288201</td>
    </tr>
    <tr>
      <th>yes</th>
      <td>0.217232</td>
      <td>1915</td>
      <td>-0.052736</td>
      <td>0.804660</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    deviceprotection
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>count</th>
      <th>diff</th>
      <th>risk</th>
    </tr>
    <tr>
      <th>deviceprotection</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>no</th>
      <td>0.395875</td>
      <td>2473</td>
      <td>0.125907</td>
      <td>1.466379</td>
    </tr>
    <tr>
      <th>no_internet_service</th>
      <td>0.077805</td>
      <td>1221</td>
      <td>-0.192163</td>
      <td>0.288201</td>
    </tr>
    <tr>
      <th>yes</th>
      <td>0.230412</td>
      <td>1940</td>
      <td>-0.039556</td>
      <td>0.853480</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    techsupport
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>count</th>
      <th>diff</th>
      <th>risk</th>
    </tr>
    <tr>
      <th>techsupport</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>no</th>
      <td>0.418914</td>
      <td>2781</td>
      <td>0.148946</td>
      <td>1.551717</td>
    </tr>
    <tr>
      <th>no_internet_service</th>
      <td>0.077805</td>
      <td>1221</td>
      <td>-0.192163</td>
      <td>0.288201</td>
    </tr>
    <tr>
      <th>yes</th>
      <td>0.159926</td>
      <td>1632</td>
      <td>-0.110042</td>
      <td>0.592390</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    streamingtv
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>count</th>
      <th>diff</th>
      <th>risk</th>
    </tr>
    <tr>
      <th>streamingtv</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>no</th>
      <td>0.342832</td>
      <td>2246</td>
      <td>0.072864</td>
      <td>1.269897</td>
    </tr>
    <tr>
      <th>no_internet_service</th>
      <td>0.077805</td>
      <td>1221</td>
      <td>-0.192163</td>
      <td>0.288201</td>
    </tr>
    <tr>
      <th>yes</th>
      <td>0.302723</td>
      <td>2167</td>
      <td>0.032755</td>
      <td>1.121328</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    streamingmovies
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>count</th>
      <th>diff</th>
      <th>risk</th>
    </tr>
    <tr>
      <th>streamingmovies</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>no</th>
      <td>0.338906</td>
      <td>2213</td>
      <td>0.068938</td>
      <td>1.255358</td>
    </tr>
    <tr>
      <th>no_internet_service</th>
      <td>0.077805</td>
      <td>1221</td>
      <td>-0.192163</td>
      <td>0.288201</td>
    </tr>
    <tr>
      <th>yes</th>
      <td>0.307273</td>
      <td>2200</td>
      <td>0.037305</td>
      <td>1.138182</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    contract
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>count</th>
      <th>diff</th>
      <th>risk</th>
    </tr>
    <tr>
      <th>contract</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>month-to-month</th>
      <td>0.431701</td>
      <td>3104</td>
      <td>0.161733</td>
      <td>1.599082</td>
    </tr>
    <tr>
      <th>one_year</th>
      <td>0.120573</td>
      <td>1186</td>
      <td>-0.149395</td>
      <td>0.446621</td>
    </tr>
    <tr>
      <th>two_year</th>
      <td>0.028274</td>
      <td>1344</td>
      <td>-0.241694</td>
      <td>0.104730</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    paperlessbilling
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>count</th>
      <th>diff</th>
      <th>risk</th>
    </tr>
    <tr>
      <th>paperlessbilling</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>no</th>
      <td>0.172071</td>
      <td>2313</td>
      <td>-0.097897</td>
      <td>0.637375</td>
    </tr>
    <tr>
      <th>yes</th>
      <td>0.338151</td>
      <td>3321</td>
      <td>0.068183</td>
      <td>1.252560</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    paymentmethod
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>count</th>
      <th>diff</th>
      <th>risk</th>
    </tr>
    <tr>
      <th>paymentmethod</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bank_transfer_(automatic)</th>
      <td>0.168171</td>
      <td>1219</td>
      <td>-0.101797</td>
      <td>0.622928</td>
    </tr>
    <tr>
      <th>credit_card_(automatic)</th>
      <td>0.164339</td>
      <td>1217</td>
      <td>-0.105630</td>
      <td>0.608733</td>
    </tr>
    <tr>
      <th>electronic_check</th>
      <td>0.455890</td>
      <td>1893</td>
      <td>0.185922</td>
      <td>1.688682</td>
    </tr>
    <tr>
      <th>mailed_check</th>
      <td>0.193870</td>
      <td>1305</td>
      <td>-0.076098</td>
      <td>0.718121</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    

## 3.6 Feature importance: Mutual information

Mutual information - concept from information theory, it tells us how much 
we can learn about one variable if we know the value of another

* https://en.wikipedia.org/wiki/Mutual_information


```python
from sklearn.metrics import mutual_info_score
```


```python
mutual_info_score(df_full_train.churn, df_full_train.contract)
```




    0.0983203874041556




```python
mutual_info_score(df_full_train.gender, df_full_train.churn)
```




    0.0001174846211139946




```python
mutual_info_score(df_full_train.contract, df_full_train.churn)
```




    0.0983203874041556




```python
mutual_info_score(df_full_train.partner, df_full_train.churn)
```




    0.009967689095399745




```python
def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)
```


```python
mi = df_full_train[categorical].apply(mutual_info_churn_score)
mi.sort_values(ascending=False)
```




    contract            0.098320
    onlinesecurity      0.063085
    techsupport         0.061032
    internetservice     0.055868
    onlinebackup        0.046923
    deviceprotection    0.043453
    paymentmethod       0.043210
    streamingtv         0.031853
    streamingmovies     0.031581
    paperlessbilling    0.017589
    dependents          0.012346
    partner             0.009968
    seniorcitizen       0.009410
    multiplelines       0.000857
    phoneservice        0.000229
    gender              0.000117
    dtype: float64



## 3.7 Feature importance: Correlation

How about numerical columns?

* Correlation coefficient - https://en.wikipedia.org/wiki/Pearson_correlation_coefficient


```python
df_full_train.tenure.max()
```




    72




```python
df_full_train[numerical].corrwith(df_full_train.churn).abs()
```




    tenure            0.351885
    monthlycharges    0.196805
    totalcharges      0.196353
    dtype: float64




```python
df_full_train[df_full_train.tenure <= 2].churn.mean()
```




    0.5953420669577875




```python
df_full_train[(df_full_train.tenure > 2) & (df_full_train.tenure <= 12)].churn.mean()
```




    0.3994413407821229




```python
df_full_train[df_full_train.tenure > 12].churn.mean()
```




    0.17634908339788277




```python
df_full_train[df_full_train.monthlycharges <= 20].churn.mean()
```




    0.08795411089866156




```python
df_full_train[(df_full_train.monthlycharges > 20) & (df_full_train.monthlycharges <= 50)].churn.mean()
```




    0.18340943683409436




```python
df_full_train[df_full_train.monthlycharges > 50].churn.mean()
```




    0.32499341585462205



## 3.8 One-hot encoding

* Use Scikit-Learn to encode categorical features


```python
from sklearn.feature_extraction import DictVectorizer
```


```python
dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)
```

## 3.9 Logistic regression

* Binary classification
* Linear vs logistic regression


```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```


```python
z = np.linspace(-7, 7, 51)
```


```python
sigmoid(10000)
```




    1.0




```python
plt.plot(z, sigmoid(z))
```




    [<matplotlib.lines.Line2D at 0x7f342d0bf080>]




    
![png](output_70_1.png)
    



```python
def linear_regression(xi):
    result = w0
    
    for j in range(len(w)):
        result = result + xi[j] * w[j]
        
    return result
```


```python
def logistic_regression(xi):
    score = w0
    
    for j in range(len(w)):
        score = score + xi[j] * w[j]
        
    result = sigmoid(score)
    return result
```

## 3.10 Training logistic regression with Scikit-Learn

* Train a model with Scikit-Learn
* Apply it to the validation dataset
* Calculate the accuracy


```python
from sklearn.linear_model import LogisticRegression
```


```python
model = LogisticRegression(solver='lbfgs')
# solver='lbfgs' is the default solver in newer version of sklearn
# for older versions, you need to specify it explicitly
model.fit(X_train, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)




```python
model.intercept_[0]
```




    -0.10903395348323511




```python
model.coef_[0].round(3)
```




    array([ 0.475, -0.175, -0.408, -0.03 , -0.078,  0.063, -0.089, -0.081,
           -0.034, -0.073, -0.335,  0.316, -0.089,  0.004, -0.258,  0.141,
            0.009,  0.063, -0.089, -0.081,  0.266, -0.089, -0.284, -0.231,
            0.124, -0.166,  0.058, -0.087, -0.032,  0.07 , -0.059,  0.141,
           -0.249,  0.215, -0.12 , -0.089,  0.102, -0.071, -0.089,  0.052,
            0.213, -0.089, -0.232, -0.07 ,  0.   ])




```python
y_pred = model.predict_proba(X_val)[:, 1]
```


```python
churn_decision = (y_pred >= 0.5)
```


```python
(y_val == churn_decision).mean()
```




    0.8034066713981547




```python
df_pred = pd.DataFrame()
df_pred['probability'] = y_pred
df_pred['prediction'] = churn_decision.astype(int)
df_pred['actual'] = y_val
```


```python
df_pred['correct'] = df_pred.prediction == df_pred.actual
```


```python
df_pred.correct.mean()
```




    0.8034066713981547




```python
churn_decision.astype(int)
```




    array([0, 0, 0, ..., 0, 1, 1])



## 3.11 Model interpretation

* Look at the coefficients
* Train a smaller model with fewer features


```python
a = [1, 2, 3, 4]
b = 'abcd'
```


```python
dict(zip(a, b))
```




    {1: 'a', 2: 'b', 3: 'c', 4: 'd'}




```python
dict(zip(dv.get_feature_names(), model.coef_[0].round(3)))
```




    {'contract=month-to-month': 0.475,
     'contract=one_year': -0.175,
     'contract=two_year': -0.408,
     'dependents=no': -0.03,
     'dependents=yes': -0.078,
     'deviceprotection=no': 0.063,
     'deviceprotection=no_internet_service': -0.089,
     'deviceprotection=yes': -0.081,
     'gender=female': -0.034,
     'gender=male': -0.073,
     'internetservice=dsl': -0.335,
     'internetservice=fiber_optic': 0.316,
     'internetservice=no': -0.089,
     'monthlycharges': 0.004,
     'multiplelines=no': -0.258,
     'multiplelines=no_phone_service': 0.141,
     'multiplelines=yes': 0.009,
     'onlinebackup=no': 0.063,
     'onlinebackup=no_internet_service': -0.089,
     'onlinebackup=yes': -0.081,
     'onlinesecurity=no': 0.266,
     'onlinesecurity=no_internet_service': -0.089,
     'onlinesecurity=yes': -0.284,
     'paperlessbilling=no': -0.231,
     'paperlessbilling=yes': 0.124,
     'partner=no': -0.166,
     'partner=yes': 0.058,
     'paymentmethod=bank_transfer_(automatic)': -0.087,
     'paymentmethod=credit_card_(automatic)': -0.032,
     'paymentmethod=electronic_check': 0.07,
     'paymentmethod=mailed_check': -0.059,
     'phoneservice=no': 0.141,
     'phoneservice=yes': -0.249,
     'seniorcitizen': 0.215,
     'streamingmovies=no': -0.12,
     'streamingmovies=no_internet_service': -0.089,
     'streamingmovies=yes': 0.102,
     'streamingtv=no': -0.071,
     'streamingtv=no_internet_service': -0.089,
     'streamingtv=yes': 0.052,
     'techsupport=no': 0.213,
     'techsupport=no_internet_service': -0.089,
     'techsupport=yes': -0.232,
     'tenure': -0.07,
     'totalcharges': 0.0}




```python
small = ['contract', 'tenure', 'monthlycharges']
```


```python
df_train[small].iloc[:10].to_dict(orient='records')
```




    [{'contract': 'two_year', 'tenure': 72, 'monthlycharges': 115.5},
     {'contract': 'month-to-month', 'tenure': 10, 'monthlycharges': 95.25},
     {'contract': 'month-to-month', 'tenure': 5, 'monthlycharges': 75.55},
     {'contract': 'month-to-month', 'tenure': 5, 'monthlycharges': 80.85},
     {'contract': 'two_year', 'tenure': 18, 'monthlycharges': 20.1},
     {'contract': 'month-to-month', 'tenure': 4, 'monthlycharges': 30.5},
     {'contract': 'month-to-month', 'tenure': 1, 'monthlycharges': 75.1},
     {'contract': 'month-to-month', 'tenure': 1, 'monthlycharges': 70.3},
     {'contract': 'two_year', 'tenure': 72, 'monthlycharges': 19.75},
     {'contract': 'month-to-month', 'tenure': 6, 'monthlycharges': 109.9}]




```python
dicts_train_small = df_train[small].to_dict(orient='records')
dicts_val_small = df_val[small].to_dict(orient='records')
```


```python
dv_small = DictVectorizer(sparse=False)
dv_small.fit(dicts_train_small)
```




    DictVectorizer(dtype=<class 'numpy.float64'>, separator='=', sort=True,
                   sparse=False)




```python
dv_small.get_feature_names()
```




    ['contract=month-to-month',
     'contract=one_year',
     'contract=two_year',
     'monthlycharges',
     'tenure']




```python
X_train_small = dv_small.transform(dicts_train_small)
```


```python
model_small = LogisticRegression(solver='lbfgs')
model_small.fit(X_train_small, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)




```python
w0 = model_small.intercept_[0]
w0
```




    -2.476775657751665




```python
w = model_small.coef_[0]
w.round(3)
```




    array([ 0.97 , -0.025, -0.949,  0.027, -0.036])




```python
dict(zip(dv_small.get_feature_names(), w.round(3)))
```




    {'contract=month-to-month': 0.97,
     'contract=one_year': -0.025,
     'contract=two_year': -0.949,
     'monthlycharges': 0.027,
     'tenure': -0.036}




```python
-2.47 + (-0.949) + 30 * 0.027 + 24 * (-0.036)
```




    -3.473




```python
sigmoid(_)
```




    0.030090303318277657



## 3.12 Using the model


```python
dicts_full_train = df_full_train[categorical + numerical].to_dict(orient='records')
```


```python
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)
```


```python
y_full_train = df_full_train.churn.values
```


```python
model = LogisticRegression(solver='lbfgs')
model.fit(X_full_train, y_full_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)




```python
dicts_test = df_test[categorical + numerical].to_dict(orient='records')
```


```python
X_test = dv.transform(dicts_test)
```


```python
y_pred = model.predict_proba(X_test)[:, 1]
```


```python
churn_decision = (y_pred >= 0.5)
```


```python
(churn_decision == y_test).mean()
```




    0.815471965933286




```python
y_test
```




    array([0, 0, 0, ..., 0, 0, 1])




```python
customer = dicts_test[-1]
customer
```




    {'gender': 'female',
     'seniorcitizen': 0,
     'partner': 'yes',
     'dependents': 'yes',
     'phoneservice': 'yes',
     'multiplelines': 'yes',
     'internetservice': 'fiber_optic',
     'onlinesecurity': 'yes',
     'onlinebackup': 'no',
     'deviceprotection': 'yes',
     'techsupport': 'no',
     'streamingtv': 'yes',
     'streamingmovies': 'yes',
     'contract': 'month-to-month',
     'paperlessbilling': 'yes',
     'paymentmethod': 'electronic_check',
     'tenure': 17,
     'monthlycharges': 104.2,
     'totalcharges': 1743.5}




```python
X_small = dv.transform([customer])
```


```python
model.predict_proba(X_small)[0, 1]
```




    0.5968852088293909




```python
y_test[-1]
```




    1


