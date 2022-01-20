<div class="cell markdown">

# 4\. Evaluation Metrics for Classification

## 4.1 Evaluation metrics: session overview

  - Dataset: <https://www.kaggle.com/blastchar/telco-customer-churn>
  - <https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv>

*Metric* - function that compares the predictions with the actual values
and outputs a single number that tells how good the predictions are

</div>

<div class="cell code" data-execution_count="1">

``` python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
```

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
df = pd.read_csv(r'E:\Zoom_mycode\Churn.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)
```

<div class="output error" data-ename="FileNotFoundError" data-evalue="[Errno 2] No such file or directory: &#39;E:\\Zoom_mycode\\Churn.csv&#39;">

    ---------------------------------------------------------------------------
    FileNotFoundError                         Traceback (most recent call last)
    <ipython-input-3-4b3cfdebd8f4> in <module>
    ----> 1 df = pd.read_csv(r'E:\Zoom_mycode\Churn.csv')
          2 
          3 df.columns = df.columns.str.lower().str.replace(' ', '_')
          4 
          5 categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
    
    ~\anaconda3\lib\site-packages\pandas\io\parsers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, dialect, error_bad_lines, warn_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options)
        608     kwds.update(kwds_defaults)
        609 
    --> 610     return _read(filepath_or_buffer, kwds)
        611 
        612 
    
    ~\anaconda3\lib\site-packages\pandas\io\parsers.py in _read(filepath_or_buffer, kwds)
        460 
        461     # Create the parser.
    --> 462     parser = TextFileReader(filepath_or_buffer, **kwds)
        463 
        464     if chunksize or iterator:
    
    ~\anaconda3\lib\site-packages\pandas\io\parsers.py in __init__(self, f, engine, **kwds)
        817             self.options["has_index_names"] = kwds["has_index_names"]
        818 
    --> 819         self._engine = self._make_engine(self.engine)
        820 
        821     def close(self):
    
    ~\anaconda3\lib\site-packages\pandas\io\parsers.py in _make_engine(self, engine)
       1048             )
       1049         # error: Too many arguments for "ParserBase"
    -> 1050         return mapping[engine](self.f, **self.options)  # type: ignore[call-arg]
       1051 
       1052     def _failover_to_python(self):
    
    ~\anaconda3\lib\site-packages\pandas\io\parsers.py in __init__(self, src, **kwds)
       1865 
       1866         # open handles
    -> 1867         self._open_handles(src, kwds)
       1868         assert self.handles is not None
       1869         for key in ("storage_options", "encoding", "memory_map", "compression"):
    
    ~\anaconda3\lib\site-packages\pandas\io\parsers.py in _open_handles(self, src, kwds)
       1360         Let the readers open IOHanldes after they are done with their potential raises.
       1361         """
    -> 1362         self.handles = get_handle(
       1363             src,
       1364             "r",
    
    ~\anaconda3\lib\site-packages\pandas\io\common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
        640                 errors = "replace"
        641             # Encoding
    --> 642             handle = open(
        643                 handle,
        644                 ioargs.mode,
    
    FileNotFoundError: [Errno 2] No such file or directory: 'E:\\Zoom_mycode\\Churn.csv'

</div>

</div>

<div class="cell code">

``` python
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

</div>

<div class="cell code">

``` python
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

</div>

<div class="cell code">

``` python
dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

model = LogisticRegression()
model.fit(X_train, y_train)
```

</div>

<div class="cell code">

``` python
val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

y_pred = model.predict_proba(X_val)[:, 1]
churn_decision = (y_pred >= 0.5)
(y_val == churn_decision).mean()
```

</div>

<div class="cell markdown">

## 4.2 Accuracy and dummy model

  - Evaluate the model on different thresholds
  - Check the accuracy of dummy baselines

</div>

<div class="cell code" data-execution_count="16">

``` python
len(y_val)
```

<div class="output execute_result" data-execution_count="16">

    1409

</div>

</div>

<div class="cell code" data-execution_count="19">

``` python
(y_val == churn_decision).mean()
```

<div class="output execute_result" data-execution_count="19">

    0.8034066713981547

</div>

</div>

<div class="cell code" data-execution_count="18">

``` python
1132/ 1409
```

<div class="output execute_result" data-execution_count="18">

    0.8034066713981547

</div>

</div>

<div class="cell code" data-execution_count="24">

``` python
from sklearn.metrics import accuracy_score
```

</div>

<div class="cell code" data-execution_count="29">

``` python
accuracy_score(y_val, y_pred >= 0.5)
```

<div class="output execute_result" data-execution_count="29">

    0.8034066713981547

</div>

</div>

<div class="cell code" data-execution_count="30">

``` python
thresholds = np.linspace(0, 1, 21)

scores = []

for t in thresholds:
    score = accuracy_score(y_val, y_pred >= t)
    print('%.2f %.3f' % (t, score))
    scores.append(score)
```

<div class="output stream stdout">

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

</div>

</div>

<div class="cell code" data-execution_count="24">

``` python
plt.plot(thresholds, scores)
```

<div class="output execute_result" data-execution_count="24">

    [<matplotlib.lines.Line2D at 0xff7b24eb48e0>]

</div>

<div class="output display_data">

![](cc20917c6a9e870f0847858eba47c0ec15729485.png)

</div>

</div>

<div class="cell code" data-execution_count="34">

``` python
from collections import Counter
```

</div>

<div class="cell code" data-execution_count="35">

``` python
Counter(y_pred >= 1.0)
```

<div class="output execute_result" data-execution_count="35">

    Counter({False: 1409})

</div>

</div>

<div class="cell code" data-execution_count="39">

``` python
1 - y_val.mean()
```

<div class="output execute_result" data-execution_count="39">

    0.7260468417317246

</div>

</div>

<div class="cell markdown">

## 4.3 Confusion table

  - Different types of errors and correct decisions
  - Arranging them in a table

</div>

<div class="cell code" data-execution_count="25">

``` python
actual_positive = (y_val == 1)
actual_negative = (y_val == 0)
```

</div>

<div class="cell code" data-execution_count="26">

``` python
t = 0.5
predict_positive = (y_pred >= t)
predict_negative = (y_pred < t)
```

</div>

<div class="cell code" data-execution_count="27">

``` python
tp = (predict_positive & actual_positive).sum()
tn = (predict_negative & actual_negative).sum()

fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()
```

</div>

<div class="cell code" data-execution_count="28">

``` python
confusion_matrix = np.array([
    [tn, fp],
    [fn, tp]
])
confusion_matrix
```

<div class="output execute_result" data-execution_count="28">

    array([[922, 101],
           [176, 210]])

</div>

</div>

<div class="cell code" data-execution_count="29">

``` python
(confusion_matrix / confusion_matrix.sum()).round(2)
```

<div class="output execute_result" data-execution_count="29">

    array([[0.65, 0.07],
           [0.12, 0.15]])

</div>

</div>

<div class="cell markdown">

## 4.4 Precision and Recall

</div>

<div class="cell code" data-execution_count="30">

``` python
p = tp / (tp + fp)
p
```

<div class="output execute_result" data-execution_count="30">

    0.6752411575562701

</div>

</div>

<div class="cell code" data-execution_count="31">

``` python
r = tp / (tp + fn)
r
```

<div class="output execute_result" data-execution_count="31">

    0.5440414507772021

</div>

</div>

<div class="cell markdown">

## 4.5 ROC Curves

### TPR and FRP

</div>

<div class="cell code" data-execution_count="32">

``` python
tpr = tp / (tp + fn)
tpr
```

<div class="output execute_result" data-execution_count="32">

    0.5440414507772021

</div>

</div>

<div class="cell code" data-execution_count="33">

``` python
fpr = fp / (fp + tn)
fpr
```

<div class="output execute_result" data-execution_count="33">

    0.09872922776148582

</div>

</div>

<div class="cell code" data-execution_count="34">

``` python
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

</div>

<div class="cell code" data-execution_count="35">

``` python
columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
df_scores = pd.DataFrame(scores, columns=columns)

df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)
```

</div>

<div class="cell code" data-execution_count="36">

``` python
plt.plot(df_scores.threshold, df_scores['tpr'], label='TPR')
plt.plot(df_scores.threshold, df_scores['fpr'], label='FPR')
plt.legend()
```

<div class="output execute_result" data-execution_count="36">

    <matplotlib.legend.Legend at 0xffaca7c6f9a0>

</div>

<div class="output display_data">

![](494f2c60cf24648028e71d560c0e28e8296ff4b3.png)

</div>

</div>

<div class="cell markdown">

### Random model

</div>

<div class="cell code" data-execution_count="37">

``` python
np.random.seed(1)
y_rand = np.random.uniform(0, 1, size=len(y_val))
```

</div>

<div class="cell code" data-execution_count="38">

``` python
((y_rand >= 0.5) == y_val).mean()
```

<div class="output execute_result" data-execution_count="38">

    0.5017743080198722

</div>

</div>

<div class="cell code" data-execution_count="47">

``` python
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

</div>

<div class="cell code" data-execution_count="40">

``` python
df_rand = tpr_fpr_dataframe(y_val, y_rand)
```

</div>

<div class="cell code" data-execution_count="41">

``` python
plt.plot(df_rand.threshold, df_rand['tpr'], label='TPR')
plt.plot(df_rand.threshold, df_rand['fpr'], label='FPR')
plt.legend()
```

<div class="output execute_result" data-execution_count="41">

    <matplotlib.legend.Legend at 0xffaca7bec8b0>

</div>

<div class="output display_data">

![](af5d2e11b3352b17f5a0dd16fa55f6a4cab4cddb.png)

</div>

</div>

<div class="cell markdown">

### Ideal model

</div>

<div class="cell code" data-execution_count="42">

``` python
num_neg = (y_val == 0).sum()
num_pos = (y_val == 1).sum()
num_neg, num_pos
```

<div class="output execute_result" data-execution_count="42">

    (1023, 386)

</div>

</div>

<div class="cell code" data-execution_count="43">

``` python

y_ideal = np.repeat([0, 1], [num_neg, num_pos])
y_ideal

y_ideal_pred = np.linspace(0, 1, len(y_val))
```

</div>

<div class="cell code" data-execution_count="44">

``` python
1 - y_val.mean()
```

<div class="output execute_result" data-execution_count="44">

    0.7260468417317246

</div>

</div>

<div class="cell code" data-execution_count="45">

``` python
accuracy_score(y_ideal, y_ideal_pred >= 0.726)
```

<div class="output execute_result" data-execution_count="45">

    1.0

</div>

</div>

<div class="cell code" data-execution_count="51">

``` python
df_ideal = tpr_fpr_dataframe(y_ideal, y_ideal_pred)
df_ideal[::10]
```

<div class="output execute_result" data-execution_count="51">

``` 
     threshold   tp    fp   fn    tn       tpr       fpr
0          0.0  386  1023    0     0  1.000000  1.000000
10         0.1  386   882    0   141  1.000000  0.862170
20         0.2  386   741    0   282  1.000000  0.724340
30         0.3  386   600    0   423  1.000000  0.586510
40         0.4  386   459    0   564  1.000000  0.448680
50         0.5  386   319    0   704  1.000000  0.311828
60         0.6  386   178    0   845  1.000000  0.173998
70         0.7  386    37    0   986  1.000000  0.036168
80         0.8  282     0  104  1023  0.730570  0.000000
90         0.9  141     0  245  1023  0.365285  0.000000
100        1.0    1     0  385  1023  0.002591  0.000000
```

</div>

</div>

<div class="cell code" data-execution_count="52">

``` python
plt.plot(df_ideal.threshold, df_ideal['tpr'], label='TPR')
plt.plot(df_ideal.threshold, df_ideal['fpr'], label='FPR')
plt.legend()
```

<div class="output execute_result" data-execution_count="52">

    <matplotlib.legend.Legend at 0xffaca7af4a90>

</div>

<div class="output display_data">

![](58ed2ccc50a5135e23b4c6e1659dfed317ccffdf.png)

</div>

</div>

<div class="cell markdown">

### Putting everything together

</div>

<div class="cell code" data-execution_count="53">

``` python
plt.plot(df_scores.threshold, df_scores['tpr'], label='TPR', color='black')
plt.plot(df_scores.threshold, df_scores['fpr'], label='FPR', color='blue')

plt.plot(df_ideal.threshold, df_ideal['tpr'], label='TPR ideal')
plt.plot(df_ideal.threshold, df_ideal['fpr'], label='FPR ideal')

# plt.plot(df_rand.threshold, df_rand['tpr'], label='TPR random', color='grey')
# plt.plot(df_rand.threshold, df_rand['fpr'], label='FPR random', color='grey')

plt.legend()
```

<div class="output execute_result" data-execution_count="53">

    <matplotlib.legend.Legend at 0xffaca7a695b0>

</div>

<div class="output display_data">

![](0084b3ed6940969a1f93a8cd8047ffaa13b5fd92.png)

</div>

</div>

<div class="cell code" data-execution_count="54">

``` python
plt.figure(figsize=(5, 5))

plt.plot(df_scores.fpr, df_scores.tpr, label='Model')
plt.plot([0, 1], [0, 1], label='Random', linestyle='--')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()
```

<div class="output execute_result" data-execution_count="54">

    <matplotlib.legend.Legend at 0xffaca6b72e50>

</div>

<div class="output display_data">

![](9fec934066aea56758161dda402a894742605ea2.png)

</div>

</div>

<div class="cell code" data-execution_count="55">

``` python
from sklearn.metrics import roc_curve
```

</div>

<div class="cell code" data-execution_count="57">

``` python
fpr, tpr, thresholds = roc_curve(y_val, y_pred)
```

</div>

<div class="cell code" data-execution_count="58">

``` python
plt.figure(figsize=(5, 5))

plt.plot(fpr, tpr, label='Model')
plt.plot([0, 1], [0, 1], label='Random', linestyle='--')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()
```

<div class="output execute_result" data-execution_count="58">

    <matplotlib.legend.Legend at 0xffaca5d3eaf0>

</div>

<div class="output display_data">

![](083ea4033ce45b3c1aaf36eb960f85585780180d.png)

</div>

</div>

<div class="cell markdown">

## 4.6 ROC AUC

  - Area under the ROC curve - useful metric
  - Interpretation of AUC

</div>

<div class="cell code" data-execution_count="60">

``` python
from sklearn.metrics import auc
```

</div>

<div class="cell code" data-execution_count="61">

``` python
auc(fpr, tpr)
```

<div class="output execute_result" data-execution_count="61">

    0.843850505725819

</div>

</div>

<div class="cell code" data-execution_count="62">

``` python
auc(df_scores.fpr, df_scores.tpr)
```

<div class="output execute_result" data-execution_count="62">

    0.8438796286447967

</div>

</div>

<div class="cell code" data-execution_count="63">

``` python
auc(df_ideal.fpr, df_ideal.tpr)
```

<div class="output execute_result" data-execution_count="63">

    0.9999430203759136

</div>

</div>

<div class="cell code" data-execution_count="68">

``` python
fpr, tpr, thresholds = roc_curve(y_val, y_pred)
auc(fpr, tpr)
```

<div class="output execute_result" data-execution_count="68">

    0.843850505725819

</div>

</div>

<div class="cell code" data-execution_count="65">

``` python
from sklearn.metrics import roc_auc_score
```

</div>

<div class="cell code" data-execution_count="66">

``` python
roc_auc_score(y_val, y_pred)
```

<div class="output execute_result" data-execution_count="66">

    0.843850505725819

</div>

</div>

<div class="cell code" data-execution_count="70">

``` python
neg = y_pred[y_val == 0]
pos = y_pred[y_val == 1]
```

</div>

<div class="cell code" data-execution_count="73">

``` python
import random
```

</div>

<div class="cell code" data-execution_count="82">

``` python
n = 100000
success = 0 

for i in range(n):
    pos_ind = random.randint(0, len(pos) - 1)
    neg_ind = random.randint(0, len(neg) - 1)

    if pos[pos_ind] > neg[neg_ind]:
        success = success + 1

success / n
```

<div class="output execute_result" data-execution_count="82">

    0.8434

</div>

</div>

<div class="cell code" data-execution_count="90">

``` python
n = 50000

np.random.seed(1)
pos_ind = np.random.randint(0, len(pos), size=n)
neg_ind = np.random.randint(0, len(neg), size=n)

(pos[pos_ind] > neg[neg_ind]).mean()
```

<div class="output execute_result" data-execution_count="90">

    0.84646

</div>

</div>

<div class="cell markdown">

## 4.7 Cross-Validation

  - Evaluating the same model on different subsets of data
  - Getting the average prediction and the spread within predictions

</div>

<div class="cell code" data-execution_count="121">

``` python
def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model
```

</div>

<div class="cell code" data-execution_count="123">

``` python
dv, model = train(df_train, y_train, C=0.001)
```

</div>

<div class="cell code" data-execution_count="110">

``` python
def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred
```

</div>

<div class="cell code" data-execution_count="98">

``` python
y_pred = predict(df_val, dv, model)
```

</div>

<div class="cell code" data-execution_count="99">

``` python
from sklearn.model_selection import KFold
```

</div>

<div class="cell code" data-execution_count="100">

``` python

```

</div>

<div class="cell code" data-execution_count="112">

``` python
!pip install tqdm
```

<div class="output stream stdout">

    Requirement already satisfied: tqdm in /home/alexey/.pyenv/versions/3.8.11/lib/python3.8/site-packages (4.61.2)
    WARNING: You are using pip version 21.2.2; however, version 21.2.4 is available.
    You should consider upgrading via the '/home/alexey/.pyenv/versions/3.8.11/bin/python3.8 -m pip install --upgrade pip' command.

</div>

</div>

<div class="cell code" data-execution_count="113">

``` python
from tqdm.auto import tqdm
```

</div>

<div class="cell code" data-execution_count="129">

``` python
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

<div class="output display_data">

``` json
{"model_id":"6bdfe85df0c7415582b9e25df38dc9c6","version_major":2,"version_minor":0}
```

</div>

<div class="output stream stdout">

    C=0.001 0.825 +- 0.009
    C=0.01 0.840 +- 0.009
    C=0.1 0.841 +- 0.008
    C=0.5 0.840 +- 0.007
    C=1 0.841 +- 0.008
    C=5 0.841 +- 0.008
    C=10 0.841 +- 0.008

</div>

</div>

<div class="cell code" data-execution_count="133">

``` python
scores
```

<div class="output execute_result" data-execution_count="133">

    [0.8419433083969826,
     0.8458047775129122,
     0.8325145494681918,
     0.8325466042079682,
     0.8525462018763139]

</div>

</div>

<div class="cell code" data-execution_count="131">

``` python
dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
auc
```

<div class="output execute_result" data-execution_count="131">

    0.8572386167896259

</div>

</div>

<div class="cell code">

``` python
```

</div>
