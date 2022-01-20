# 4. Evaluation Metrics for Classification

In the previous session we trained a model for predicting churn. How do we know if it's good?


## 4.1 Evaluation metrics: session overview 

* Dataset: https://www.kaggle.com/blastchar/telco-customer-churn
* https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv


*Metric* - function that compares the predictions with the actual values and outputs a single number that tells how good the predictions are


```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
```


```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
```


```python
df = pd.read_csv(r'E:\Zoom_mycode\Churn\churn.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)
```


```python
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']
```


```python
numerical = ['tenure', 'monthlycharges', 'totalcharges']

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
dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

model = LogisticRegression()
model.fit(X_train, y_train)
```


```python
val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

y_pred = model.predict_proba(X_val)[:, 1]
churn_decision = (y_pred >= 0.5)
(y_val == churn_decision).mean()
```

## 4.2 Accuracy and dummy model

* Evaluate the model on different thresholds
* Check the accuracy of dummy baselines


```python
len(y_val)
```




    1409




```python
(y_val == churn_decision).mean()
```




    0.8034066713981547




```python
1132/ 1409
```




    0.8034066713981547




```python
from sklearn.metrics import accuracy_score
```


```python
accuracy_score(y_val, y_pred >= 0.5)
```




    0.8034066713981547




```python
thresholds = np.linspace(0, 1, 21)

scores = []

for t in thresholds:
    score = accuracy_score(y_val, y_pred >= t)
    print('%.2f %.3f' % (t, score))
    scores.append(score)
```

    0.00 0.274
    0.05 0.509
    0.10 0.591
    0.15 0.666
    0.20 0.710
    0.25 0.739
    0.30 0.760
    0.35 0.772
    0.40 0.785
    0.45 0.793
    0.50 0.803
    0.55 0.801
    0.60 0.795
    0.65 0.786
    0.70 0.766
    0.75 0.744
    0.80 0.735
    0.85 0.726
    0.90 0.726
    0.95 0.726
    1.00 0.726
    


```python
plt.plot(thresholds, scores)
```




    [<matplotlib.lines.Line2D at 0xff7b24eb48e0>]




    
![png](output_15_1.png)
    



```python
from collections import Counter
```


```python
Counter(y_pred >= 1.0)
```




    Counter({False: 1409})




```python
1 - y_val.mean()
```




    0.7260468417317246



## 4.3 Confusion table

* Different types of errors and correct decisions
* Arranging them in a table


```python
actual_positive = (y_val == 1)
actual_negative = (y_val == 0)
```


```python
t = 0.5
predict_positive = (y_pred >= t)
predict_negative = (y_pred < t)
```


```python
tp = (predict_positive & actual_positive).sum()
tn = (predict_negative & actual_negative).sum()

fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()
```


```python
confusion_matrix = np.array([
    [tn, fp],
    [fn, tp]
])
confusion_matrix
```




    array([[922, 101],
           [176, 210]])




```python
(confusion_matrix / confusion_matrix.sum()).round(2)
```




    array([[0.65, 0.07],
           [0.12, 0.15]])



## 4.4 Precision and Recall


```python
p = tp / (tp + fp)
p
```




    0.6752411575562701




```python
r = tp / (tp + fn)
r
```




    0.5440414507772021



## 4.5 ROC Curves

### TPR and FRP


```python
tpr = tp / (tp + fn)
tpr
```




    0.5440414507772021




```python
fpr = fp / (fp + tn)
fpr
```




    0.09872922776148582




```python
scores = []

thresholds = np.linspace(0, 1, 101)

for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)
    
    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()

    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()
    
    scores.append((t, tp, fp, fn, tn))
```


```python
columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
df_scores = pd.DataFrame(scores, columns=columns)

df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)
```


```python
plt.plot(df_scores.threshold, df_scores['tpr'], label='TPR')
plt.plot(df_scores.threshold, df_scores['fpr'], label='FPR')
plt.legend()
```




    <matplotlib.legend.Legend at 0xffaca7c6f9a0>




    
![png](output_33_1.png)
    


### Random model


```python
np.random.seed(1)
y_rand = np.random.uniform(0, 1, size=len(y_val))
```


```python
((y_rand >= 0.5) == y_val).mean()
```




    0.5017743080198722




```python
def tpr_fpr_dataframe(y_val, y_pred):
    scores = []

    thresholds = np.linspace(0, 1, 101)

    for t in thresholds:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)

        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = (predict_positive & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()

        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()

        scores.append((t, tp, fp, fn, tn))

    columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
    df_scores = pd.DataFrame(scores, columns=columns)

    df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)
    
    return df_scores
```


```python
df_rand = tpr_fpr_dataframe(y_val, y_rand)
```


```python
plt.plot(df_rand.threshold, df_rand['tpr'], label='TPR')
plt.plot(df_rand.threshold, df_rand['fpr'], label='FPR')
plt.legend()
```




    <matplotlib.legend.Legend at 0xffaca7bec8b0>




    
![png](output_39_1.png)
    


### Ideal model


```python
num_neg = (y_val == 0).sum()
num_pos = (y_val == 1).sum()
num_neg, num_pos
```




    (1023, 386)




```python

y_ideal = np.repeat([0, 1], [num_neg, num_pos])
y_ideal

y_ideal_pred = np.linspace(0, 1, len(y_val))
```


```python
1 - y_val.mean()
```




    0.7260468417317246




```python
accuracy_score(y_ideal, y_ideal_pred >= 0.726)
```




    1.0




```python
df_ideal = tpr_fpr_dataframe(y_ideal, y_ideal_pred)
df_ideal[::10]
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
      <th>threshold</th>
      <th>tp</th>
      <th>fp</th>
      <th>fn</th>
      <th>tn</th>
      <th>tpr</th>
      <th>fpr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>386</td>
      <td>1023</td>
      <td>0</td>
      <td>0</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.1</td>
      <td>386</td>
      <td>882</td>
      <td>0</td>
      <td>141</td>
      <td>1.000000</td>
      <td>0.862170</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.2</td>
      <td>386</td>
      <td>741</td>
      <td>0</td>
      <td>282</td>
      <td>1.000000</td>
      <td>0.724340</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.3</td>
      <td>386</td>
      <td>600</td>
      <td>0</td>
      <td>423</td>
      <td>1.000000</td>
      <td>0.586510</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0.4</td>
      <td>386</td>
      <td>459</td>
      <td>0</td>
      <td>564</td>
      <td>1.000000</td>
      <td>0.448680</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.5</td>
      <td>386</td>
      <td>319</td>
      <td>0</td>
      <td>704</td>
      <td>1.000000</td>
      <td>0.311828</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.6</td>
      <td>386</td>
      <td>178</td>
      <td>0</td>
      <td>845</td>
      <td>1.000000</td>
      <td>0.173998</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.7</td>
      <td>386</td>
      <td>37</td>
      <td>0</td>
      <td>986</td>
      <td>1.000000</td>
      <td>0.036168</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.8</td>
      <td>282</td>
      <td>0</td>
      <td>104</td>
      <td>1023</td>
      <td>0.730570</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>90</th>
      <td>0.9</td>
      <td>141</td>
      <td>0</td>
      <td>245</td>
      <td>1023</td>
      <td>0.365285</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>385</td>
      <td>1023</td>
      <td>0.002591</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.plot(df_ideal.threshold, df_ideal['tpr'], label='TPR')
plt.plot(df_ideal.threshold, df_ideal['fpr'], label='FPR')
plt.legend()
```




    <matplotlib.legend.Legend at 0xffaca7af4a90>




    
![png](output_46_1.png)
    


### Putting everything together


```python
plt.plot(df_scores.threshold, df_scores['tpr'], label='TPR', color='black')
plt.plot(df_scores.threshold, df_scores['fpr'], label='FPR', color='blue')

plt.plot(df_ideal.threshold, df_ideal['tpr'], label='TPR ideal')
plt.plot(df_ideal.threshold, df_ideal['fpr'], label='FPR ideal')

# plt.plot(df_rand.threshold, df_rand['tpr'], label='TPR random', color='grey')
# plt.plot(df_rand.threshold, df_rand['fpr'], label='FPR random', color='grey')

plt.legend()
```




    <matplotlib.legend.Legend at 0xffaca7a695b0>




    
![png](output_48_1.png)
    



```python
plt.figure(figsize=(5, 5))

plt.plot(df_scores.fpr, df_scores.tpr, label='Model')
plt.plot([0, 1], [0, 1], label='Random', linestyle='--')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()
```




    <matplotlib.legend.Legend at 0xffaca6b72e50>




    
![png](output_49_1.png)
    



```python
from sklearn.metrics import roc_curve
```


```python
fpr, tpr, thresholds = roc_curve(y_val, y_pred)
```


```python
plt.figure(figsize=(5, 5))

plt.plot(fpr, tpr, label='Model')
plt.plot([0, 1], [0, 1], label='Random', linestyle='--')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()
```




    <matplotlib.legend.Legend at 0xffaca5d3eaf0>




    
![png](output_52_1.png)
    


## 4.6 ROC AUC

* Area under the ROC curve - useful metric
* Interpretation of AUC


```python
from sklearn.metrics import auc
```


```python
auc(fpr, tpr)
```




    0.843850505725819




```python
auc(df_scores.fpr, df_scores.tpr)
```




    0.8438796286447967




```python
auc(df_ideal.fpr, df_ideal.tpr)
```




    0.9999430203759136




```python
fpr, tpr, thresholds = roc_curve(y_val, y_pred)
auc(fpr, tpr)
```




    0.843850505725819




```python
from sklearn.metrics import roc_auc_score
```


```python
roc_auc_score(y_val, y_pred)
```




    0.843850505725819




```python
neg = y_pred[y_val == 0]
pos = y_pred[y_val == 1]
```


```python
import random
```


```python
n = 100000
success = 0 

for i in range(n):
    pos_ind = random.randint(0, len(pos) - 1)
    neg_ind = random.randint(0, len(neg) - 1)

    if pos[pos_ind] > neg[neg_ind]:
        success = success + 1

success / n
```




    0.8434




```python
n = 50000

np.random.seed(1)
pos_ind = np.random.randint(0, len(pos), size=n)
neg_ind = np.random.randint(0, len(neg), size=n)

(pos[pos_ind] > neg[neg_ind]).mean()
```




    0.84646



## 4.7 Cross-Validation

* Evaluating the same model on different subsets of data
* Getting the average prediction and the spread within predictions


```python
def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model
```


```python
dv, model = train(df_train, y_train, C=0.001)
```


```python
def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred
```


```python
y_pred = predict(df_val, dv, model)
```


```python
from sklearn.model_selection import KFold
```


```python


```


```python
!pip install tqdm
```

    Requirement already satisfied: tqdm in /home/alexey/.pyenv/versions/3.8.11/lib/python3.8/site-packages (4.61.2)
    [33mWARNING: You are using pip version 21.2.2; however, version 21.2.4 is available.
    You should consider upgrading via the '/home/alexey/.pyenv/versions/3.8.11/bin/python3.8 -m pip install --upgrade pip' command.[0m
    


```python
from tqdm.auto import tqdm
```


```python
n_splits = 5

for C in tqdm([0.001, 0.01, 0.1, 0.5, 1, 5, 10]):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.churn.values
        y_val = df_val.churn.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))
```


      0%|          | 0/7 [00:00<?, ?it/s]


    C=0.001 0.825 +- 0.009
    C=0.01 0.840 +- 0.009
    C=0.1 0.841 +- 0.008
    C=0.5 0.840 +- 0.007
    C=1 0.841 +- 0.008
    C=5 0.841 +- 0.008
    C=10 0.841 +- 0.008
    


```python
scores
```




    [0.8419433083969826,
     0.8458047775129122,
     0.8325145494681918,
     0.8325466042079682,
     0.8525462018763139]




```python
dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
auc
```




    0.8572386167896259




```python

```
