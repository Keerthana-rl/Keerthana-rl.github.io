<div class="cell code" data-execution_count="1">

``` python
import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline
```

</div>

<div class="cell markdown">

Let's train the model again first - to use its results later in this
notebook

</div>

<div class="cell code" data-execution_count="2">

``` python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
```

</div>

<div class="cell code" data-execution_count="3">

``` python
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

df.churn = (df.churn == 'yes').astype(int)
```

</div>

<div class="cell code" data-execution_count="4">

``` python
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_train_full, test_size=0.33, random_state=11)

y_train = df_train.churn.values
y_val = df_val.churn.values

del df_train['churn']
del df_val['churn']
```

</div>

<div class="cell code" data-execution_count="5">

``` python
categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies',
               'contract', 'paperlessbilling', 'paymentmethod']
numerical = ['tenure', 'monthlycharges', 'totalcharges']
```

</div>

<div class="cell code" data-execution_count="6" data-scrolled="false">

``` python
train_dict = df_train[categorical + numerical].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

X_train = dv.transform(train_dict)
```

</div>

<div class="cell code" data-execution_count="7">

``` python
model = LogisticRegression(solver='liblinear', random_state=1)
model.fit(X_train, y_train)
```

<div class="output execute_result" data-execution_count="7">

    LogisticRegression(random_state=1, solver='liblinear')

</div>

</div>

<div class="cell code" data-execution_count="8">

``` python
val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)
y_pred = model.predict_proba(X_val)[:, 1]
```

</div>

<div class="cell code" data-execution_count="9">

``` python
small_subset = ['contract', 'tenure', 'totalcharges']
train_dict_small = df_train[small_subset].to_dict(orient='records')
dv_small = DictVectorizer(sparse=False)
dv_small.fit(train_dict_small)

X_small_train = dv_small.transform(train_dict_small)

model_small = LogisticRegression(solver='liblinear', random_state=1)
model_small.fit(X_small_train, y_train)
```

<div class="output execute_result" data-execution_count="9">

    LogisticRegression(random_state=1, solver='liblinear')

</div>

</div>

<div class="cell code" data-execution_count="10">

``` python
val_dict_small = df_val[small_subset].to_dict(orient='records')
X_small_val = dv_small.transform(val_dict_small)

y_pred_small = model_small.predict_proba(X_small_val)[:, 1]
```

</div>

<div class="cell markdown">

## Accuracy

</div>

<div class="cell code" data-execution_count="11">

``` python
y_pred = model.predict_proba(X_val)[:, 1]
churn = y_pred >= 0.5
(churn == y_val).mean()
```

<div class="output execute_result" data-execution_count="11">

    0.8016129032258065

</div>

</div>

<div class="cell code" data-execution_count="12">

``` python
from sklearn.metrics import accuracy_score
```

</div>

<div class="cell code" data-execution_count="13">

``` python
accuracy_score(y_val, y_pred >= 0.5)
```

<div class="output execute_result" data-execution_count="13">

    0.8016129032258065

</div>

</div>

<div class="cell code" data-execution_count="14">

``` python
thresholds = np.linspace(0, 1, 11)
thresholds
```

<div class="output execute_result" data-execution_count="14">

    array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])

</div>

</div>

<div class="cell code" data-execution_count="15">

``` python
thresholds = np.linspace(0, 1, 21)

accuracies = []

for t in thresholds:
    acc = accuracy_score(y_val, y_pred >= t)
    accuracies.append(acc)
    print('%0.2f %0.3f' % (t, acc))
```

<div class="output stream stdout">

    0.00 0.261
    0.05 0.501
    0.10 0.595
    0.15 0.640
    0.20 0.690
    0.25 0.730
    0.30 0.755
    0.35 0.767
    0.40 0.782
    0.45 0.795
    0.50 0.802
    0.55 0.790
    0.60 0.790
    0.65 0.788
    0.70 0.774
    0.75 0.752
    0.80 0.742
    0.85 0.739
    0.90 0.739
    0.95 0.739
    1.00 0.739

</div>

</div>

<div class="cell code" data-execution_count="16">

``` python
plt.figure(figsize=(6, 4))

plt.plot(thresholds, accuracies, color='black')

plt.title('Threshold vs Accuracy')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')

plt.xticks(np.linspace(0, 1, 11))

# plt.savefig('04_threshold_accuracy.svg')

plt.show()
```

<div class="output display_data">

![](2a6e119ace41cb998a2e1d2102b684ea9f2e3aa8.png)

</div>

</div>

<div class="cell code" data-execution_count="17">

``` python
churn_small = y_pred_small >= 0.5
(churn_small == y_val).mean()
```

<div class="output execute_result" data-execution_count="17">

    0.7672043010752688

</div>

</div>

<div class="cell code" data-execution_count="18">

``` python
accuracy_score(y_val, churn_small)
```

<div class="output execute_result" data-execution_count="18">

    0.7672043010752688

</div>

</div>

<div class="cell code" data-execution_count="19" data-scrolled="true">

``` python
size_val = len(y_val)
baseline = np.repeat(False, size_val)
baseline
```

<div class="output execute_result" data-execution_count="19">

    array([False, False, False, ..., False, False, False])

</div>

</div>

<div class="cell code" data-execution_count="20">

``` python
accuracy_score(baseline, y_val)
```

<div class="output execute_result" data-execution_count="20">

    0.7387096774193549

</div>

</div>

<div class="cell markdown">

## Confusion table

</div>

<div class="cell code" data-execution_count="21">

``` python
true_positive = ((y_pred >= 0.5) & (y_val == 1)).sum()
false_positive = ((y_pred >= 0.5) & (y_val == 0)).sum()
false_negative = ((y_pred < 0.5) & (y_val == 1)).sum()
true_negative = ((y_pred < 0.5) & (y_val == 0)).sum()
```

</div>

<div class="cell code" data-execution_count="22" data-scrolled="true">

``` python
confusion_table = np.array(
     # predict neg    pos
    [[true_negative, false_positive], # actual neg
     [false_negative, true_positive]]) # actual pos

confusion_table
```

<div class="output execute_result" data-execution_count="22">

    array([[1202,  172],
           [ 197,  289]])

</div>

</div>

<div class="cell code" data-execution_count="23" data-scrolled="false">

``` python
confusion_table / confusion_table.sum()
```

<div class="output execute_result" data-execution_count="23">

    array([[0.64623656, 0.09247312],
           [0.10591398, 0.15537634]])

</div>

</div>

<div class="cell markdown">

## Precision and recall

</div>

<div class="cell code" data-execution_count="24">

``` python
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
precision, recall
```

<div class="output execute_result" data-execution_count="24">

    (0.6268980477223427, 0.5946502057613169)

</div>

</div>

<div class="cell code" data-execution_count="25">

``` python
confusion_table / confusion_table.sum()
```

<div class="output execute_result" data-execution_count="25">

    array([[0.64623656, 0.09247312],
           [0.10591398, 0.15537634]])

</div>

</div>

<div class="cell code" data-execution_count="26">

``` python
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
precision, recall
```

<div class="output execute_result" data-execution_count="26">

    (0.6268980477223427, 0.5946502057613169)

</div>

</div>

<div class="cell markdown">

## ROC and AUC

</div>

<div class="cell markdown">

TPR and FPR

</div>

<div class="cell code" data-execution_count="27">

``` python
scores = []

thresholds = np.linspace(0, 1, 101)

for t in thresholds: #B
    tp = ((y_pred >= t) & (y_val == 1)).sum()
    fp = ((y_pred >= t) & (y_val == 0)).sum()
    fn = ((y_pred < t) & (y_val == 1)).sum()
    tn = ((y_pred < t) & (y_val == 0)).sum()
    scores.append((t, tp, fp, fn, tn))

df_scores = pd.DataFrame(scores)
df_scores.columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
```

</div>

<div class="cell code" data-execution_count="28">

``` python
df_scores[::10]
```

<div class="output execute_result" data-execution_count="28">

``` 
     threshold   tp    fp   fn    tn
0          0.0  486  1374    0     0
10         0.1  458   726   28   648
20         0.2  421   512   65   862
30         0.3  380   350  106  1024
40         0.4  337   257  149  1117
50         0.5  289   172  197  1202
60         0.6  200   105  286  1269
70         0.7   99    34  387  1340
80         0.8    7     1  479  1373
90         0.9    0     0  486  1374
100        1.0    0     0  486  1374
```

</div>

</div>

<div class="cell code" data-execution_count="29">

``` python
df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)
```

</div>

<div class="cell code" data-execution_count="30">

``` python
df_scores[::10]
```

<div class="output execute_result" data-execution_count="30">

``` 
     threshold   tp    fp   fn    tn       tpr       fpr
0          0.0  486  1374    0     0  1.000000  1.000000
10         0.1  458   726   28   648  0.942387  0.528384
20         0.2  421   512   65   862  0.866255  0.372635
30         0.3  380   350  106  1024  0.781893  0.254731
40         0.4  337   257  149  1117  0.693416  0.187045
50         0.5  289   172  197  1202  0.594650  0.125182
60         0.6  200   105  286  1269  0.411523  0.076419
70         0.7   99    34  387  1340  0.203704  0.024745
80         0.8    7     1  479  1373  0.014403  0.000728
90         0.9    0     0  486  1374  0.000000  0.000000
100        1.0    0     0  486  1374  0.000000  0.000000
```

</div>

</div>

<div class="cell code" data-execution_count="31">

``` python
plt.figure(figsize=(6, 4))

plt.plot(df_scores.threshold, df_scores.tpr, color='black', linestyle='solid', label='TPR')
plt.plot(df_scores.threshold, df_scores.fpr, color='black', linestyle='dashed', label='FPR')
plt.legend()

plt.xticks(np.linspace(0, 1, 11))
plt.yticks(np.linspace(0, 1, 11))

plt.xlabel('Thresholds')
plt.title('TPR and FPR')

# plt.savefig('04_fpr_tpr_plot.svg')

plt.show()
```

<div class="output display_data">

![](f3ab997e7485cfa216513fff29a9577a1461e2b5.png)

</div>

</div>

<div class="cell markdown">

Random baseline

</div>

<div class="cell code" data-execution_count="32">

``` python
def tpr_fpr_dataframe(y_val, y_pred):
    scores = []

    thresholds = np.linspace(0, 1, 101)

    for t in thresholds:
        tp = ((y_pred >= t) & (y_val == 1)).sum()
        fp = ((y_pred >= t) & (y_val == 0)).sum()
        fn = ((y_pred < t) & (y_val == 1)).sum()
        tn = ((y_pred < t) & (y_val == 0)).sum()

        scores.append((t, tp, fp, fn, tn))

    df_scores = pd.DataFrame(scores)
    df_scores.columns = ['threshold', 'tp', 'fp', 'fn', 'tn']

    df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)

    return df_scores
```

</div>

<div class="cell code" data-execution_count="33">

``` python
np.random.seed(1)
y_rand = np.random.uniform(0, 1, size=len(y_val))
df_rand = tpr_fpr_dataframe(y_val, y_rand)
df_rand[::10]
```

<div class="output execute_result" data-execution_count="33">

``` 
     threshold   tp    fp   fn    tn       tpr       fpr
0          0.0  486  1374    0     0  1.000000  1.000000
10         0.1  440  1236   46   138  0.905350  0.899563
20         0.2  392  1101   94   273  0.806584  0.801310
30         0.3  339   972  147   402  0.697531  0.707424
40         0.4  288   849  198   525  0.592593  0.617904
50         0.5  239   723  247   651  0.491770  0.526201
60         0.6  193   579  293   795  0.397119  0.421397
70         0.7  152   422  334   952  0.312757  0.307132
80         0.8   98   302  388  1072  0.201646  0.219796
90         0.9   57   147  429  1227  0.117284  0.106987
100        1.0    0     0  486  1374  0.000000  0.000000
```

</div>

</div>

<div class="cell code" data-execution_count="34" data-scrolled="true">

``` python
plt.figure(figsize=(6, 4))

plt.plot(df_rand.threshold, df_rand.tpr, color='black', linestyle='solid', label='TPR')
plt.plot(df_rand.threshold, df_rand.fpr, color='black', linestyle='dashed', label='FPR')
plt.legend()

plt.xticks(np.linspace(0, 1, 11))
plt.yticks(np.linspace(0, 1, 11))

plt.xlabel('Thresholds')
plt.title('TPR and FPR for the random model')

#plt.savefig('04_fpr_tpr_plot_random.svg')

plt.show()
```

<div class="output display_data">

![](a7e3abfbb099698928359484b473bf5d52d7f336.png)

</div>

</div>

<div class="cell markdown">

Ideal baseline:

</div>

<div class="cell code" data-execution_count="35">

``` python
num_neg = (y_val == 0).sum()
num_pos = (y_val == 1).sum()

y_ideal = np.repeat([0, 1], [num_neg, num_pos])
y_pred_ideal = np.linspace(0, 1, num_neg + num_pos)

df_ideal = tpr_fpr_dataframe(y_ideal, y_pred_ideal)
df_ideal[::10]
```

<div class="output execute_result" data-execution_count="35">

``` 
     threshold   tp    fp   fn    tn       tpr       fpr
0          0.0  486  1374    0     0  1.000000  1.000000
10         0.1  486  1188    0   186  1.000000  0.864629
20         0.2  486  1002    0   372  1.000000  0.729258
30         0.3  486   816    0   558  1.000000  0.593886
40         0.4  486   630    0   744  1.000000  0.458515
50         0.5  486   444    0   930  1.000000  0.323144
60         0.6  486   258    0  1116  1.000000  0.187773
70         0.7  486    72    0  1302  1.000000  0.052402
80         0.8  372     0  114  1374  0.765432  0.000000
90         0.9  186     0  300  1374  0.382716  0.000000
100        1.0    1     0  485  1374  0.002058  0.000000
```

</div>

</div>

<div class="cell code" data-execution_count="36" data-scrolled="true">

``` python
plt.figure(figsize=(6, 4))

plt.plot(df_ideal.threshold, df_ideal.tpr, color='black', linestyle='solid', label='TPR')
plt.plot(df_ideal.threshold, df_ideal.fpr, color='black', linestyle='dashed', label='FPR')
plt.legend()

plt.xticks(np.linspace(0, 1, 11))
plt.yticks(np.linspace(0, 1, 11))

plt.vlines(1 - y_val.mean(), -1, 2, linewidth=0.5, linestyle='dashed', color='grey')
plt.ylim(-0.03, 1.03)

plt.xlabel('Thresholds')
plt.title('TPR and FPR for the ideal model')

# plt.savefig('04_fpr_tpr_plot_ideal.svg')

plt.show()
```

<div class="output display_data">

![](a3a8939a2d3b381d5a73e9cd2a359e8d08a21799.png)

</div>

</div>

<div class="cell markdown">

ROC curve

</div>

<div class="cell code" data-execution_count="37">

``` python
plt.figure(figsize=(5, 5))

plt.plot(df_scores.fpr, df_scores.tpr, color='black', label='Model')
plt.plot(df_rand.fpr, df_rand.tpr, color='black', lw=1,
         linestyle='dashed', alpha=0.5, label='Random')
plt.plot(df_ideal.fpr, df_ideal.tpr, color='black', lw=0.5,
         linestyle='solid', alpha=0.5, label='Ideal')

plt.legend()

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('ROC curve')

# plt.savefig('04_roc_curve_with_baselines.svg')

plt.show()
```

<div class="output display_data">

![](e5dfbdb6bdab26ad292563b0f0f565ac41d09dff.png)

</div>

</div>

<div class="cell code" data-execution_count="38">

``` python
plt.figure(figsize=(5, 5))

plt.plot(df_scores.fpr, df_scores.tpr, color='black')
plt.plot([0, 1], [0, 1], color='black', lw=0.7, linestyle='dashed', alpha=0.5)

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('ROC curve')

# plt.savefig('04_roc_curve.svg')

plt.show()
```

<div class="output display_data">

![](bc1f15607e25037a2e506b04ed9de864b6e0594c.png)

</div>

</div>

<div class="cell markdown">

Using Scikit-Learn for plotting the ROC curve

</div>

<div class="cell code" data-execution_count="39">

``` python
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
```

</div>

<div class="cell code" data-execution_count="40">

``` python
fpr, tpr, thresholds = roc_curve(y_val, y_pred)
```

</div>

<div class="cell code" data-execution_count="41">

``` python
plt.figure(figsize=(5, 5))

plt.plot(fpr, tpr, color='black')
plt.plot([0, 1], [0, 1], color='black', lw=0.7, linestyle='dashed', alpha=0.5)

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('ROC curve')

plt.show()
```

<div class="output display_data">

![](9680cc4b1ac4c74ab9e301a138c31dd28e60f6f8.png)

</div>

</div>

<div class="cell markdown">

AUC: Area under the ROC curve

</div>

<div class="cell code" data-execution_count="42">

``` python
df_scores_small = tpr_fpr_dataframe(y_val, y_pred_small)
```

</div>

<div class="cell code" data-execution_count="43" data-scrolled="true">

``` python
auc(df_scores.fpr, df_scores.tpr)
```

<div class="output execute_result" data-execution_count="43">

    0.8359150837721111

</div>

</div>

<div class="cell code" data-execution_count="44" data-scrolled="false">

``` python
auc(df_scores_small.fpr, df_scores_small.tpr)
```

<div class="output execute_result" data-execution_count="44">

    0.8108718050089552

</div>

</div>

<div class="cell markdown">

Comparing multiple models with ROC curves

</div>

<div class="cell code" data-execution_count="45">

``` python
fpr_large, tpr_large, _ = roc_curve(y_val, y_pred)
fpr_small, tpr_small, _ = roc_curve(y_val, y_pred_small)

plt.figure(figsize=(5, 5))

plt.plot(fpr_large, tpr_large, color='black', linestyle='solid', label='Large')
plt.plot(fpr_small, tpr_small, color='black', linestyle='dashed', label='Small')
plt.plot([0, 1], [0, 1], color='black', lw=0.7, linestyle='dashed', alpha=0.5)

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('ROC curve')
plt.legend(loc='lower right')

plt.show()
```

<div class="output display_data">

![](6edf8685af88465e3e58fd88906eb6367b348679.png)

</div>

</div>

<div class="cell code" data-execution_count="46">

``` python
from sklearn.metrics import roc_auc_score
```

</div>

<div class="cell code" data-execution_count="47">

``` python
roc_auc_score(y_val, y_pred)
```

<div class="output execute_result" data-execution_count="47">

    0.8363381374257972

</div>

</div>

<div class="cell code" data-execution_count="48">

``` python
roc_auc_score(y_val, y_pred_small)
```

<div class="output execute_result" data-execution_count="48">

    0.8117942866042492

</div>

</div>

<div class="cell markdown">

Interpretation of AUC: the probability that a randomly chosen positive
example ranks higher than a randomly chosen negative example

</div>

<div class="cell code" data-execution_count="49" data-scrolled="false">

``` python
neg = y_pred[y_val == 0]
pos = y_pred[y_val == 1]

np.random.seed(1)
neg_choice = np.random.randint(low=0, high=len(neg), size=10000)
pos_choice = np.random.randint(low=0, high=len(pos), size=10000)
(pos[pos_choice] > neg[neg_choice]).mean()
```

<div class="output execute_result" data-execution_count="49">

    0.8356

</div>

</div>

<div class="cell markdown">

## K-fold cross-validation

</div>

<div class="cell code" data-execution_count="50">

``` python
def train(df, y):
    cat = df[categorical + numerical].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    dv.fit(cat)

    X = dv.transform(cat)

    model = LogisticRegression(solver='liblinear')
    model.fit(X, y)

    return dv, model


def predict(df, dv, model):
    cat = df[categorical + numerical].to_dict(orient='records')
    
    X = dv.transform(cat)

    y_pred = model.predict_proba(X)[:, 1]

    return y_pred
```

</div>

<div class="cell code" data-execution_count="51">

``` python
from sklearn.model_selection import KFold
```

</div>

<div class="cell code" data-execution_count="52" data-scrolled="false">

``` python
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
```

</div>

<div class="cell code" data-execution_count="53">

``` python
aucs = []

for train_idx, val_idx in kfold.split(df_train_full):
    df_train = df_train_full.iloc[train_idx]
    y_train = df_train.churn.values

    df_val = df_train_full.iloc[val_idx]
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train)
    y_pred = predict(df_val, dv, model)

    rocauc = roc_auc_score(y_val, y_pred)
    aucs.append(rocauc)
```

</div>

<div class="cell code" data-execution_count="54">

``` python
np.array(aucs).round(3)
```

<div class="output execute_result" data-execution_count="54">

    array([0.849, 0.841, 0.859, 0.833, 0.824, 0.842, 0.844, 0.822, 0.845,
           0.861])

</div>

</div>

<div class="cell code" data-execution_count="55" data-scrolled="true">

``` python
print('auc = %0.3f ± %0.3f' % (np.mean(aucs), np.std(aucs)))
```

<div class="output stream stdout">

    auc = 0.842 ± 0.012

</div>

</div>

<div class="cell markdown">

Tuning the parameter `C`

</div>

<div class="cell code" data-execution_count="56">

``` python
def train(df, y, C=1.0):
    cat = df[categorical + numerical].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    dv.fit(cat)

    X = dv.transform(cat)

    model = LogisticRegression(solver='liblinear', C=C)
    model.fit(X, y)

    return dv, model
```

</div>

<div class="cell code" data-execution_count="57" data-scrolled="true">

``` python
nfolds = 5
kfold = KFold(n_splits=nfolds, shuffle=True, random_state=1)

for C in [0.001, 0.01, 0.1, 0.5, 1, 10]:
    aucs = []

    for train_idx, val_idx in kfold.split(df_train_full):
        df_train = df_train_full.iloc[train_idx]
        df_val = df_train_full.iloc[val_idx]

        y_train = df_train.churn.values
        y_val = df_val.churn.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)
        
        auc = roc_auc_score(y_val, y_pred)
        aucs.append(auc)

    print('C=%s, auc = %0.3f ± %0.3f' % (C, np.mean(aucs), np.std(aucs)))
```

<div class="output stream stdout">

    C=0.001, auc = 0.825 ± 0.013
    C=0.01, auc = 0.839 ± 0.009
    C=0.1, auc = 0.841 ± 0.007
    C=0.5, auc = 0.841 ± 0.007
    C=1, auc = 0.841 ± 0.007
    C=10, auc = 0.841 ± 0.007

</div>

</div>

<div class="cell markdown">

Full retrain

</div>

<div class="cell code" data-execution_count="58">

``` python
y_train = df_train_full.churn.values
y_test = df_test.churn.values

dv, model = train(df_train_full, y_train, C=0.5)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
print('auc = %.3f' % auc)
```

<div class="output stream stdout">

    auc = 0.858

</div>

</div>

<div class="cell code">

``` python
```

</div>
