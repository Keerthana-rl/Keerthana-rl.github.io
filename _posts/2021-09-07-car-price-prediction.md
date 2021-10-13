---
title:  "Week 2 - ML Zoom Camp"
permalink: /posts/lr-normal-equation/
excerpt: "Week-2 covers the simple linear regression using Normal Equation from Scratch"
last_modified_at: 2021-09-19T16:00:11-04:00
header:
  #image: assets/images/car.jpg
  teaser: assets/images/car.jpg
categories:
- tutorial
tags:
- mlzoomcamp
toc: true
toc_sticky: true
#classes: wide
---

# Car Price Prediction Project

In this week, we are going to learn our first ML algorithm regression through a machine learning project. The dataset can be download from [here](https://www.kaggle.com/CooperUnion/cardataset). We have many features like ***Make, Year, City MPG*** and we have a target feature ***MSRP***

## Project Plan

* Prepare data and do EDA
* Use linear regression for predicting price
* Understanding internals of linear regression (Vector form) and implementing it
* Evaluating model with RMSE
* Feature Engineering
* Regularization

## Data Preparation


```python
import numpy as np
import pandas as pd
```


```python
# Uncomment below line to download dataset
# !wget https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv
```


```python
df = pd.read_csv('data.csv')
df.head() # looking first few rows to understand data
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Market Category</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BMW</td>
      <td>1 Series M</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>335.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Factory Tuner,Luxury,High-Performance</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>26</td>
      <td>19</td>
      <td>3916</td>
      <td>46135</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury,Performance</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>28</td>
      <td>19</td>
      <td>3916</td>
      <td>40650</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury,High-Performance</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>28</td>
      <td>20</td>
      <td>3916</td>
      <td>36350</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>230.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury,Performance</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>28</td>
      <td>18</td>
      <td>3916</td>
      <td>29450</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>230.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>28</td>
      <td>18</td>
      <td>3916</td>
      <td>34500</td>
    </tr>
  </tbody>
</table>
</div>



### Observations from .head()

- Column names either separated by white space or hypen
- Column names either small or capital letters
- Values in the data either small or capital letters

The columns must be cleaned and put everything in same order


```python
df.columns # display column names in the data
```




    Index(['Make', 'Model', 'Year', 'Engine Fuel Type', 'Engine HP',
           'Engine Cylinders', 'Transmission Type', 'Driven_Wheels',
           'Number of Doors', 'Market Category', 'Vehicle Size', 'Vehicle Style',
           'highway MPG', 'city mpg', 'Popularity', 'MSRP'],
          dtype='object')




```python
# data cleaning on column names
df.columns = df.columns.str.lower().str.replace(" ","_")
# converting the strings (column names) into lowercase and then replace & fill whitespace with '-'
df.columns
```




    Index(['make', 'model', 'year', 'engine_fuel_type', 'engine_hp',
           'engine_cylinders', 'transmission_type', 'driven_wheels',
           'number_of_doors', 'market_category', 'vehicle_size', 'vehicle_style',
           'highway_mpg', 'city_mpg', 'popularity', 'msrp'],
          dtype='object')




```python
df.dtypes # display data types of each column
```




    make                  object
    model                 object
    year                   int64
    engine_fuel_type      object
    engine_hp            float64
    engine_cylinders     float64
    transmission_type     object
    driven_wheels         object
    number_of_doors      float64
    market_category       object
    vehicle_size          object
    vehicle_style         object
    highway_mpg            int64
    city_mpg               int64
    popularity             int64
    msrp                   int64
    dtype: object




```python
strings = list(df.dtypes[df.dtypes == 'object'].index)
strings
```




    ['make',
     'model',
     'engine_fuel_type',
     'transmission_type',
     'driven_wheels',
     'market_category',
     'vehicle_size',
     'vehicle_style']



- **[df.dtypes == 'object']** shows column with true if dtype is object
- **df.dtypes[df.dtypes == 'object']** shows only columns with object datatypes
- **df.dtypes[df.dtypes == 'object'].index** shows column as indexes
- **list(df.dtypes[df.dtypes == 'object'].index)** converts indexes as list


```python
for column in strings:
    df[column] = df[column].str.lower().str.replace(" ","_") 
    # exactly how we did for column names
```


```python
df.select_dtypes("object").head() # display only object datatypes
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
      <th>make</th>
      <th>model</th>
      <th>engine_fuel_type</th>
      <th>transmission_type</th>
      <th>driven_wheels</th>
      <th>market_category</th>
      <th>vehicle_size</th>
      <th>vehicle_style</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bmw</td>
      <td>1_series_m</td>
      <td>premium_unleaded_(required)</td>
      <td>manual</td>
      <td>rear_wheel_drive</td>
      <td>factory_tuner,luxury,high-performance</td>
      <td>compact</td>
      <td>coupe</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bmw</td>
      <td>1_series</td>
      <td>premium_unleaded_(required)</td>
      <td>manual</td>
      <td>rear_wheel_drive</td>
      <td>luxury,performance</td>
      <td>compact</td>
      <td>convertible</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bmw</td>
      <td>1_series</td>
      <td>premium_unleaded_(required)</td>
      <td>manual</td>
      <td>rear_wheel_drive</td>
      <td>luxury,high-performance</td>
      <td>compact</td>
      <td>coupe</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bmw</td>
      <td>1_series</td>
      <td>premium_unleaded_(required)</td>
      <td>manual</td>
      <td>rear_wheel_drive</td>
      <td>luxury,performance</td>
      <td>compact</td>
      <td>coupe</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bmw</td>
      <td>1_series</td>
      <td>premium_unleaded_(required)</td>
      <td>manual</td>
      <td>rear_wheel_drive</td>
      <td>luxury</td>
      <td>compact</td>
      <td>convertible</td>
    </tr>
  </tbody>
</table>
</div>



## Explanatory Data Analysis (EDA)


```python
for column in df.columns:
    print(column) # display column name
    print('-'*10)
    print(df[column].unique()[:5]) # display first 5 unique values of each column 
    print(df[column].nunique())
    print()
```

    make
    ----------
    ['bmw' 'audi' 'fiat' 'mercedes-benz' 'chrysler']
    48
    
    model
    ----------
    ['1_series_m' '1_series' '100' '124_spider' '190-class']
    914
    
    year
    ----------
    [2011 2012 2013 1992 1993]
    28
    
    engine_fuel_type
    ----------
    ['premium_unleaded_(required)' 'regular_unleaded'
     'premium_unleaded_(recommended)' 'flex-fuel_(unleaded/e85)' 'diesel']
    10
    
    engine_hp
    ----------
    [335. 300. 230. 320. 172.]
    356
    
    engine_cylinders
    ----------
    [ 6.  4.  5.  8. 12.]
    9
    
    transmission_type
    ----------
    ['manual' 'automatic' 'automated_manual' 'direct_drive' 'unknown']
    5
    
    driven_wheels
    ----------
    ['rear_wheel_drive' 'front_wheel_drive' 'all_wheel_drive'
     'four_wheel_drive']
    4
    
    number_of_doors
    ----------
    [ 2.  4.  3. nan]
    3
    
    market_category
    ----------
    ['factory_tuner,luxury,high-performance' 'luxury,performance'
     'luxury,high-performance' 'luxury' 'performance']
    71
    
    vehicle_size
    ----------
    ['compact' 'midsize' 'large']
    3
    
    vehicle_style
    ----------
    ['coupe' 'convertible' 'sedan' 'wagon' '4dr_hatchback']
    16
    
    highway_mpg
    ----------
    [26 28 27 25 24]
    59
    
    city_mpg
    ----------
    [19 20 18 17 16]
    69
    
    popularity
    ----------
    [3916 3105  819  617 1013]
    48
    
    msrp
    ----------
    [46135 40650 36350 29450 34500]
    6049
    



```python
df.describe() # by default describe shows for numeric values
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
      <th>year</th>
      <th>engine_hp</th>
      <th>engine_cylinders</th>
      <th>number_of_doors</th>
      <th>highway_mpg</th>
      <th>city_mpg</th>
      <th>popularity</th>
      <th>msrp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>11914.000000</td>
      <td>11845.00000</td>
      <td>11884.000000</td>
      <td>11908.000000</td>
      <td>11914.000000</td>
      <td>11914.000000</td>
      <td>11914.000000</td>
      <td>1.191400e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2010.384338</td>
      <td>249.38607</td>
      <td>5.628829</td>
      <td>3.436093</td>
      <td>26.637485</td>
      <td>19.733255</td>
      <td>1554.911197</td>
      <td>4.059474e+04</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.579740</td>
      <td>109.19187</td>
      <td>1.780559</td>
      <td>0.881315</td>
      <td>8.863001</td>
      <td>8.987798</td>
      <td>1441.855347</td>
      <td>6.010910e+04</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1990.000000</td>
      <td>55.00000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>12.000000</td>
      <td>7.000000</td>
      <td>2.000000</td>
      <td>2.000000e+03</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2007.000000</td>
      <td>170.00000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>22.000000</td>
      <td>16.000000</td>
      <td>549.000000</td>
      <td>2.100000e+04</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2015.000000</td>
      <td>227.00000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>26.000000</td>
      <td>18.000000</td>
      <td>1385.000000</td>
      <td>2.999500e+04</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2016.000000</td>
      <td>300.00000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>30.000000</td>
      <td>22.000000</td>
      <td>2009.000000</td>
      <td>4.223125e+04</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2017.000000</td>
      <td>1001.00000</td>
      <td>16.000000</td>
      <td>4.000000</td>
      <td>354.000000</td>
      <td>137.000000</td>
      <td>5657.000000</td>
      <td>2.065902e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe(include=[object]) # show statistics for categorical values
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
      <th>make</th>
      <th>model</th>
      <th>engine_fuel_type</th>
      <th>transmission_type</th>
      <th>driven_wheels</th>
      <th>market_category</th>
      <th>vehicle_size</th>
      <th>vehicle_style</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>11914</td>
      <td>11914</td>
      <td>11911</td>
      <td>11914</td>
      <td>11914</td>
      <td>8172</td>
      <td>11914</td>
      <td>11914</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>48</td>
      <td>914</td>
      <td>10</td>
      <td>5</td>
      <td>4</td>
      <td>71</td>
      <td>3</td>
      <td>16</td>
    </tr>
    <tr>
      <th>top</th>
      <td>chevrolet</td>
      <td>silverado_1500</td>
      <td>regular_unleaded</td>
      <td>automatic</td>
      <td>front_wheel_drive</td>
      <td>crossover</td>
      <td>compact</td>
      <td>sedan</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1123</td>
      <td>156</td>
      <td>7172</td>
      <td>8266</td>
      <td>4787</td>
      <td>1110</td>
      <td>4764</td>
      <td>3048</td>
    </tr>
  </tbody>
</table>
</div>



## Describing Price


```python
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = [12,10]

%matplotlib inline
```


```python
sns.histplot(df.msrp, bins=10);
```


    
![plot-1](https://raw.githubusercontent.com/edwardcodes/edwardcodes.github.io/main/assets/images/week-2/output.png)
    


Based on the above figure, we can observe the prices (msrp) values are higher which would confuse our model


```python
sns.histplot(df.msrp[df.msrp < 100000], bins=50);
```


    
![plot-2](https://raw.githubusercontent.com/edwardcodes/edwardcodes.github.io/main/assets/images/week-2/output-2.png)
    


### Logarithmic function on values

To avoid, we apply logarithmic function on msrp value so that, even higher values become smaller. Check the below example


```python
np.log([1,10,1000,10000,100000])
```




    array([ 0.        ,  2.30258509,  6.90775528,  9.21034037, 11.51292546])




```python
# but if we apply log on zero, values becomes inifity
np.log(0)
```

    /tmp/ipykernel_19857/1310345647.py:2: RuntimeWarning: divide by zero encountered in log
      np.log(0)





    -inf




```python
# to avoid that we have to add 1 if we have zero values np.log(0 + 1) ~= np.log(1) = 0
# but numpy has inbuilt function np.log1p which adds 1 automically
np.log1p(0)
```




    0.0




```python
# Coming back to our datasets, we dont have zero msrp values
price_logs = np.log1p(df.msrp)
price_logs
```




    0        10.739349
    1        10.612779
    2        10.500977
    3        10.290483
    4        10.448744
               ...    
    11909    10.739024
    11910    10.945018
    11911    10.832122
    11912    10.838031
    11913    10.274913
    Name: msrp, Length: 11914, dtype: float64




```python
# applying hsitogram on price_logs
sns.histplot(price_logs, bins=50);
```


    
![plot-3](https://raw.githubusercontent.com/edwardcodes/edwardcodes.github.io/main/assets/images/week-2/output-3.png)
    


## Missing values


```python
df.isnull().sum().sort_values(ascending=False)
```




    market_category      3742
    engine_hp              69
    engine_cylinders       30
    number_of_doors         6
    engine_fuel_type        3
    make                    0
    model                   0
    year                    0
    transmission_type       0
    driven_wheels           0
    vehicle_size            0
    vehicle_style           0
    highway_mpg             0
    city_mpg                0
    popularity              0
    msrp                    0
    dtype: int64



## Setting validation framework


```python
# we are splitting data into 60% for training, 20% for validation and remaining 20% for testing
n = len(df)

n_val = int(0.2 * n) #20% validation
n_test = int(0.2 * n) #20% testing
n_train = n - (n_test + n_val)
```


```python
n_train, n_test, n_val
```




    (7150, 2382, 2382)




```python
# Separating dataframe based on above ratio
df_train = df.iloc[:n_train]
df_val = df.iloc[n_train:n_val+n_train] #[:2382]
df_test = df.iloc[n_val+n_train:] #[2382:4764]

```


```python
df_test
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
      <th>make</th>
      <th>model</th>
      <th>year</th>
      <th>engine_fuel_type</th>
      <th>engine_hp</th>
      <th>engine_cylinders</th>
      <th>transmission_type</th>
      <th>driven_wheels</th>
      <th>number_of_doors</th>
      <th>market_category</th>
      <th>vehicle_size</th>
      <th>vehicle_style</th>
      <th>highway_mpg</th>
      <th>city_mpg</th>
      <th>popularity</th>
      <th>msrp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9532</th>
      <td>chevrolet</td>
      <td>silverado_1500</td>
      <td>2015</td>
      <td>regular_unleaded</td>
      <td>355.0</td>
      <td>8.0</td>
      <td>automatic</td>
      <td>rear_wheel_drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>large</td>
      <td>crew_cab_pickup</td>
      <td>23</td>
      <td>16</td>
      <td>1385</td>
      <td>47575</td>
    </tr>
    <tr>
      <th>9533</th>
      <td>chevrolet</td>
      <td>silverado_1500</td>
      <td>2015</td>
      <td>flex-fuel_(unleaded/e85)</td>
      <td>285.0</td>
      <td>6.0</td>
      <td>automatic</td>
      <td>rear_wheel_drive</td>
      <td>4.0</td>
      <td>flex_fuel</td>
      <td>large</td>
      <td>extended_cab_pickup</td>
      <td>24</td>
      <td>18</td>
      <td>1385</td>
      <td>31940</td>
    </tr>
    <tr>
      <th>9534</th>
      <td>chevrolet</td>
      <td>silverado_1500</td>
      <td>2015</td>
      <td>flex-fuel_(unleaded/e85)</td>
      <td>285.0</td>
      <td>6.0</td>
      <td>automatic</td>
      <td>rear_wheel_drive</td>
      <td>4.0</td>
      <td>flex_fuel</td>
      <td>large</td>
      <td>crew_cab_pickup</td>
      <td>24</td>
      <td>18</td>
      <td>1385</td>
      <td>38335</td>
    </tr>
    <tr>
      <th>9535</th>
      <td>chevrolet</td>
      <td>silverado_1500</td>
      <td>2015</td>
      <td>flex-fuel_(unleaded/e85)</td>
      <td>285.0</td>
      <td>6.0</td>
      <td>automatic</td>
      <td>rear_wheel_drive</td>
      <td>4.0</td>
      <td>flex_fuel</td>
      <td>large</td>
      <td>extended_cab_pickup</td>
      <td>24</td>
      <td>18</td>
      <td>1385</td>
      <td>35870</td>
    </tr>
    <tr>
      <th>9536</th>
      <td>chevrolet</td>
      <td>silverado_1500</td>
      <td>2015</td>
      <td>flex-fuel_(unleaded/e85)</td>
      <td>285.0</td>
      <td>6.0</td>
      <td>automatic</td>
      <td>rear_wheel_drive</td>
      <td>2.0</td>
      <td>flex_fuel</td>
      <td>large</td>
      <td>regular_cab_pickup</td>
      <td>24</td>
      <td>18</td>
      <td>1385</td>
      <td>28155</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11909</th>
      <td>acura</td>
      <td>zdx</td>
      <td>2012</td>
      <td>premium_unleaded_(required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>automatic</td>
      <td>all_wheel_drive</td>
      <td>4.0</td>
      <td>crossover,hatchback,luxury</td>
      <td>midsize</td>
      <td>4dr_hatchback</td>
      <td>23</td>
      <td>16</td>
      <td>204</td>
      <td>46120</td>
    </tr>
    <tr>
      <th>11910</th>
      <td>acura</td>
      <td>zdx</td>
      <td>2012</td>
      <td>premium_unleaded_(required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>automatic</td>
      <td>all_wheel_drive</td>
      <td>4.0</td>
      <td>crossover,hatchback,luxury</td>
      <td>midsize</td>
      <td>4dr_hatchback</td>
      <td>23</td>
      <td>16</td>
      <td>204</td>
      <td>56670</td>
    </tr>
    <tr>
      <th>11911</th>
      <td>acura</td>
      <td>zdx</td>
      <td>2012</td>
      <td>premium_unleaded_(required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>automatic</td>
      <td>all_wheel_drive</td>
      <td>4.0</td>
      <td>crossover,hatchback,luxury</td>
      <td>midsize</td>
      <td>4dr_hatchback</td>
      <td>23</td>
      <td>16</td>
      <td>204</td>
      <td>50620</td>
    </tr>
    <tr>
      <th>11912</th>
      <td>acura</td>
      <td>zdx</td>
      <td>2013</td>
      <td>premium_unleaded_(recommended)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>automatic</td>
      <td>all_wheel_drive</td>
      <td>4.0</td>
      <td>crossover,hatchback,luxury</td>
      <td>midsize</td>
      <td>4dr_hatchback</td>
      <td>23</td>
      <td>16</td>
      <td>204</td>
      <td>50920</td>
    </tr>
    <tr>
      <th>11913</th>
      <td>lincoln</td>
      <td>zephyr</td>
      <td>2006</td>
      <td>regular_unleaded</td>
      <td>221.0</td>
      <td>6.0</td>
      <td>automatic</td>
      <td>front_wheel_drive</td>
      <td>4.0</td>
      <td>luxury</td>
      <td>midsize</td>
      <td>sedan</td>
      <td>26</td>
      <td>17</td>
      <td>61</td>
      <td>28995</td>
    </tr>
  </tbody>
</table>
<p>2382 rows × 16 columns</p>
</div>



**But the problem in doing above, Our model would memorize everything as we split dataset in sequential manner, So we have to shuffle data and our model find patterns instead of memorizing, which won't be overfitting**


```python
idx = np.arange(n) # capturing the indexes of dataframe
idx
```




    array([    0,     1,     2, ..., 11911, 11912, 11913])




```python
np.random.seed(2)
np.random.shuffle(idx)
```


```python
idx # shuffled randomly
```




    array([2735, 6720, 5878, ..., 6637, 2575, 7336])




```python
idx[:n_train] # use the same for val and test in dataframe
```




    array([2735, 6720, 5878, ..., 9334, 5284, 2420])




```python
df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_val+n_train]] # [:2382]
df_test = df.iloc[idx[n_val+n_train:]] # [2382:4764]
```


```python
df_train.head()
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
      <th>make</th>
      <th>model</th>
      <th>year</th>
      <th>engine_fuel_type</th>
      <th>engine_hp</th>
      <th>engine_cylinders</th>
      <th>transmission_type</th>
      <th>driven_wheels</th>
      <th>number_of_doors</th>
      <th>market_category</th>
      <th>vehicle_size</th>
      <th>vehicle_style</th>
      <th>highway_mpg</th>
      <th>city_mpg</th>
      <th>popularity</th>
      <th>msrp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2735</th>
      <td>chevrolet</td>
      <td>cobalt</td>
      <td>2008</td>
      <td>regular_unleaded</td>
      <td>148.0</td>
      <td>4.0</td>
      <td>manual</td>
      <td>front_wheel_drive</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>compact</td>
      <td>coupe</td>
      <td>33</td>
      <td>24</td>
      <td>1385</td>
      <td>14410</td>
    </tr>
    <tr>
      <th>6720</th>
      <td>toyota</td>
      <td>matrix</td>
      <td>2012</td>
      <td>regular_unleaded</td>
      <td>132.0</td>
      <td>4.0</td>
      <td>automatic</td>
      <td>front_wheel_drive</td>
      <td>4.0</td>
      <td>hatchback</td>
      <td>compact</td>
      <td>4dr_hatchback</td>
      <td>32</td>
      <td>25</td>
      <td>2031</td>
      <td>19685</td>
    </tr>
    <tr>
      <th>5878</th>
      <td>subaru</td>
      <td>impreza</td>
      <td>2016</td>
      <td>regular_unleaded</td>
      <td>148.0</td>
      <td>4.0</td>
      <td>automatic</td>
      <td>all_wheel_drive</td>
      <td>4.0</td>
      <td>hatchback</td>
      <td>compact</td>
      <td>4dr_hatchback</td>
      <td>37</td>
      <td>28</td>
      <td>640</td>
      <td>19795</td>
    </tr>
    <tr>
      <th>11190</th>
      <td>volkswagen</td>
      <td>vanagon</td>
      <td>1991</td>
      <td>regular_unleaded</td>
      <td>90.0</td>
      <td>4.0</td>
      <td>manual</td>
      <td>rear_wheel_drive</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>large</td>
      <td>passenger_minivan</td>
      <td>18</td>
      <td>16</td>
      <td>873</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>4554</th>
      <td>ford</td>
      <td>f-150</td>
      <td>2017</td>
      <td>flex-fuel_(unleaded/e85)</td>
      <td>385.0</td>
      <td>8.0</td>
      <td>automatic</td>
      <td>four_wheel_drive</td>
      <td>4.0</td>
      <td>flex_fuel</td>
      <td>large</td>
      <td>crew_cab_pickup</td>
      <td>21</td>
      <td>15</td>
      <td>5657</td>
      <td>56260</td>
    </tr>
  </tbody>
</table>
</div>




```python
# we have to convert our y values as seen before
np.log1p([1,10,1000,10000000])
```




    array([ 0.69314718,  2.39789527,  6.90875478, 16.11809575])




```python
y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)
```


```python
# after separating target values, deleting target variables from train, val, test dataframe 
del df_train['msrp']
del df_test['msrp']
del df_val['msrp']
```

# Linear Regression

### Example


```python
df_train.iloc[10]
```




    make                                 rolls-royce
    model                     phantom_drophead_coupe
    year                                        2015
    engine_fuel_type     premium_unleaded_(required)
    engine_hp                                  453.0
    engine_cylinders                            12.0
    transmission_type                      automatic
    driven_wheels                   rear_wheel_drive
    number_of_doors                              2.0
    market_category        exotic,luxury,performance
    vehicle_size                               large
    vehicle_style                        convertible
    highway_mpg                                   19
    city_mpg                                      11
    popularity                                    86
    Name: 7557, dtype: object




```python
xi = [453,11,86] # example values of one feature from our dataset
w0 = 5.17 # random bias
w = [.01, .04, .002]
```


```python
def linear_regression(xi):
    """Simple Linear Regression"""
    n_loops = len(xi)

    pred = w0 #initial weight for bias

    for j in range(n_loops):
        pred += w[j] * xi[j]
    return pred
```


```python
print(linear_regression(xi))
# Note that it produces lower value due to we are going to apply log on y values
```

    10.312



```python
# To get the exact values, we have to reverse the log values by exponential
np.expm1(10.312) # compare this predicted price with 'y' value
```




    30090.55961642849




```python
np.log1p(30090.55961642849)
```




    10.312



## Linear Regression in Vector Form

![Simple LR](https://raw.githubusercontent.com/edwardcodes/edwardcodes.github.io/main/assets/images/week-2/simple-lr.jpeg)


```python
def dot_mat(xi,w):
    """Dot Product of xi vector"""
    n_runs = len(xi)

    result = 0.0

    for run in range(n_runs):
        result += xi[run] * w[run]
    return result
```


```python
[w0] + w # w0 = 5.17
```




    [5.17, 0.01, 0.04, 0.002]




```python
[1] + xi # xi0 =1
```




    [1, 453, 11, 86]




```python
def lr_form(xi):
    w_new = [w0] + w 
    xi = [1] + xi
    return dot_mat(xi,w_new)
```


```python
print(lr_form(xi))
```

    10.312


### Example - With Multiple features


```python
w0 = 5.17
w_new = [w0] + w
w_new
```




    [5.17, 0.01, 0.04, 0.002]




```python
# three sample features
x1  = [1, 148, 24, 1385]
x2  = [1, 132, 25, 2031]
x10 = [1, 453, 11, 86]

X = [x1, x2, x10]
X = np.array(X)
X
```




    array([[   1,  148,   24, 1385],
           [   1,  132,   25, 2031],
           [   1,  453,   11,   86]])




```python
def lr_multiple_features(xi,w):
    return xi.dot(w)

print(lr_multiple_features(xi=X,w=w_new))
```

    [10.38  11.552 10.312]


***

## Training Linear Regression


```python
# Multiple features
X = [
    [148, 24, 1385],
    [132, 25, 2031],
    [453, 11, 86],
    [158, 24, 185],
    [172, 25, 201],
    [413, 11, 86],
    [38,  54, 185],
    [142, 25, 431],
    [453, 31, 86],
]
X = np.array(X)
X
```




    array([[ 148,   24, 1385],
           [ 132,   25, 2031],
           [ 453,   11,   86],
           [ 158,   24,  185],
           [ 172,   25,  201],
           [ 413,   11,   86],
           [  38,   54,  185],
           [ 142,   25,  431],
           [ 453,   31,   86]])




```python
np.set_printoptions(suppress=True) # roundoff value instead of showng values 1e+07 to normal number
```


```python
# creating bias for X, keeping bias would help us to understand baseline price of the car
ones = np.ones(X.shape[0])
ones
```




    array([1., 1., 1., 1., 1., 1., 1., 1., 1.])




```python
# append ones values in X column wise for considering ones as bias
X = np.column_stack([ones,X])
X
```




    array([[   1.,  148.,   24., 1385.],
           [   1.,  132.,   25., 2031.],
           [   1.,  453.,   11.,   86.],
           [   1.,  158.,   24.,  185.],
           [   1.,  172.,   25.,  201.],
           [   1.,  413.,   11.,   86.],
           [   1.,   38.,   54.,  185.],
           [   1.,  142.,   25.,  431.],
           [   1.,  453.,   31.,   86.]])




```python
# y values
y = [10000, 20000, 15000, 20050, 10000, 20000, 15000, 25000, 12000]
y
```




    [10000, 20000, 15000, 20050, 10000, 20000, 15000, 25000, 12000]




```python
XTX = X.T.dot(X)
XTX
```




    array([[      9.,    2109.,     230.,    4676.],
           [   2109.,  696471.,   44115.,  718540.],
           [    230.,   44115.,    7146.,  118803.],
           [   4676.,  718540.,  118803., 6359986.]])




```python
XTX_inv = np.linalg.inv(XTX)
XTX_inv
```




    array([[ 3.30686958, -0.00539612, -0.06213256, -0.00066102],
           [-0.00539612,  0.00001116,  0.0000867 ,  0.00000109],
           [-0.06213256,  0.0000867 ,  0.00146189,  0.00000858],
           [-0.00066102,  0.00000109,  0.00000858,  0.00000036]])




```python
X.T
```




    array([[   1.,    1.,    1.,    1.,    1.,    1.,    1.,    1.,    1.],
           [ 148.,  132.,  453.,  158.,  172.,  413.,   38.,  142.,  453.],
           [  24.,   25.,   11.,   24.,   25.,   11.,   54.,   25.,   31.],
           [1385., 2031.,   86.,  185.,  201.,   86.,  185.,  431.,   86.]])




```python
# putting together everything to calculate W
w_full = XTX_inv.dot(X.T).dot(y)
w_full
```




    array([25844.75405577,   -16.08906468,  -199.47254894,    -1.22802883])




```python
# Separating w0 and other weights (w)
w0 = w_full[0]
w = w_full[1:]
w0, w
```




    (25844.75405576679, array([ -16.08906468, -199.47254894,   -1.22802883]))




```python
X
```




    array([[   1.,  148.,   24., 1385.],
           [   1.,  132.,   25., 2031.],
           [   1.,  453.,   11.,   86.],
           [   1.,  158.,   24.,  185.],
           [   1.,  172.,   25.,  201.],
           [   1.,  413.,   11.,   86.],
           [   1.,   38.,   54.,  185.],
           [   1.,  142.,   25.,  431.],
           [   1.,  453.,   31.,   86.]])



### Putting together everything


```python
X = [
    [148, 24, 1385],
    [132, 25, 2031],
    [453, 11, 86],
    [158, 24, 185],
    [172, 25, 201],
    [413, 11, 86],
    [38,  54, 185],
    [142, 25, 431],
    [453, 31, 86],
]
X = np.array(X)
X
```




    array([[ 148,   24, 1385],
           [ 132,   25, 2031],
           [ 453,   11,   86],
           [ 158,   24,  185],
           [ 172,   25,  201],
           [ 413,   11,   86],
           [  38,   54,  185],
           [ 142,   25,  431],
           [ 453,   31,   86]])




```python
y
```




    [10000, 20000, 15000, 20050, 10000, 20000, 15000, 25000, 12000]




```python
# putting everything inside function - DRY (Dont Repeat Yourself)
def linear_regression(X,y):
    """Calculate linear regression"""
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones,X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]
    
```


```python
print(linear_regression(X,y))
```

    (25844.75405576679, array([ -16.08906468, -199.47254894,   -1.22802883]))


***

## Car price baseline model


```python
df_train.columns
```




    Index(['make', 'model', 'year', 'engine_fuel_type', 'engine_hp',
           'engine_cylinders', 'transmission_type', 'driven_wheels',
           'number_of_doors', 'market_category', 'vehicle_size', 'vehicle_style',
           'highway_mpg', 'city_mpg', 'popularity'],
          dtype='object')




```python
# selecting few features to create baseline model
base = ['engine_hp', 'engine_cylinders', 'highway_mpg',
        'city_mpg', 'popularity']

# before creating baseline model, check for missing values
df_train[base].isnull().sum().sort_values(ascending=False)
```




    engine_hp           40
    engine_cylinders    14
    highway_mpg          0
    city_mpg             0
    popularity           0
    dtype: int64




```python
# fill missing values with '0' for our understanding
X_train = df_train[base].fillna(0).values # .values convert dataframe to numpy array
X_train
```




    array([[ 148.,    4.,   33.,   24., 1385.],
           [ 132.,    4.,   32.,   25., 2031.],
           [ 148.,    4.,   37.,   28.,  640.],
           ...,
           [ 285.,    6.,   22.,   17.,  549.],
           [ 563.,   12.,   21.,   13.,   86.],
           [ 200.,    4.,   31.,   22.,  873.]])




```python
# calculate base and weights with help of our custom function
w0, w = linear_regression(X_train,y_train)
```


```python
w0
```




    7.927257388070117




```python
w
```




    array([ 0.0097059 , -0.15910349,  0.01437921,  0.01494411, -0.00000907])




```python
# Calculating predictions based on base and weights
y_pred = w0 + X_train.dot(w)
y_pred
```




    array([ 9.54792783,  9.38733977,  9.67197758, ..., 10.30423015,
           11.9778914 ,  9.99863111])




```python
# Plot and compare our results
sns.histplot(y_pred, alpha = 0.5, color='red', bins=50) # alpha for transparency
sns.histplot(y_train, alpha = 0.5, color='pink', bins=50);
```


    
![plot-4](https://raw.githubusercontent.com/edwardcodes/edwardcodes.github.io/main/assets/images/week-2/output-4.png)
    


**As we check from above figure, our y_pred predicts lower value for higher output values**

## RMSE


```python
def rmse(y,y_pred):
    """Calculate RMSE value of a model"""
    sq_error = (y - y_pred)**2
    mean_sq_error = sq_error.mean()
    return np.sqrt(mean_sq_error)
```


```python
rmse(y_train,y_pred)
```




    0.7554192603920132



## Validating the Model on Validation Dataset

**Based on earlier data preparation steps below, we are going to create function**
1. base = ['engine_hp', 'engine_cylinders', 'highway_mpg','city_mpg', 'popularity']
2. df_train[base].isnull().sum().sort_values(ascending=False)
3. X_train = df_train[base].fillna(0).values 
4. w0, w = linear_regression(X_train,y_train)
5. y_pred = w0 + X_train.dot(w)


```python
# from earlier selected features
base = ['engine_hp', 'engine_cylinders', 'highway_mpg','city_mpg', 'popularity'] 
```


```python
def prepare_X(df):
    df_num = df[base] # selecting features from df
    df_num = df_num.fillna(0) # filling missing values with 0
    X = df_num.values # converting dataframe to NumPy array
    return X
```


```python
# From earlier written linear regression
def linear_regression(X,y):
    """Calculate linear regression"""
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones,X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]
    
```


```python
X_train = prepare_X(df_train) # preparing X_train dataset after cleaning missing values
w0, w = linear_regression(X_train,y_train) # produces weights of base and parameters

X_val = prepare_X(df_val) # Preparing validation dataset to compare with train dataset
y_pred = w0 + X_val.dot(w) # applying known values of w0, w from train dataset

# Comparing the evaluated validation prediction with existing validation target values
rmse(y_val,y_pred) 
```




    0.7616530991301601



***

## Ways to improve

### Simple Feature Engineering

**Car's Age plays important role in features and we missed in our baseline**


```python
df_train.columns
```




    Index(['make', 'model', 'year', 'engine_fuel_type', 'engine_hp',
           'engine_cylinders', 'transmission_type', 'driven_wheels',
           'number_of_doors', 'market_category', 'vehicle_size', 'vehicle_style',
           'highway_mpg', 'city_mpg', 'popularity'],
          dtype='object')




```python
df_train["year"].max() # max. collected year of car is 2017
```




    2017




```python
2017 - df_train.year
```




    2735      9
    6720      5
    5878      1
    11190    26
    4554      0
             ..
    434       2
    1902      2
    9334      2
    5284      3
    2420      0
    Name: year, Length: 7150, dtype: int64




```python
# Lets add this feature for preparing our train data
def prepare_X(df):
    # copying just to make sure adding/modifying new features shouldn't affect original data
    df = df.copy()
    df['age'] = 2017 - df['year'] # creating new feature in copied dataset
    features = base + ['age'] # adding 'age' feature with existing features in base list

    df_num = df[features] # selecting features from df
    df_num = df_num.fillna(0) # filling missing values with 0
    X = df_num.values # converting dataframe to NumPy array
    return X
```


```python
X_train = prepare_X(df_train) # preparing X_train dataset after cleaning missing values
w0, w = linear_regression(X_train,y_train) # produces weights of base and parameters

X_val = prepare_X(df_val) # Preparing validation dataset to compare with train dataset
y_pred = w0 + X_val.dot(w) # applying known values of w0, w from train dataset

# Comparing the evaluated validation prediction with existing validation target values
rmse(y_val,y_pred)
```




    0.5172055461058335



**As we checked with our earlier RMSE value, it decreased from 0.76 to 0.51 which is a better model. Adding important features making model prediction approx., equal to actual target values**


```python
# Lets plot to visualize the impact of adding new feature
sns.histplot(y_pred, label='prediction', color='red', alpha=0.5, bins=50)
sns.histplot(y_val, label='target', color='blue',  alpha=0.5, bins=50)
plt.legend();
```


    
![plot-5](https://raw.githubusercontent.com/edwardcodes/edwardcodes.github.io/main/assets/images/week-2/output-5.png)
    


### Categorical Variables

#### Number Of Doors


```python
df_train.dtypes
```




    make                  object
    model                 object
    year                   int64
    engine_fuel_type      object
    engine_hp            float64
    engine_cylinders     float64
    transmission_type     object
    driven_wheels         object
    number_of_doors      float64
    market_category       object
    vehicle_size          object
    vehicle_style         object
    highway_mpg            int64
    city_mpg               int64
    popularity             int64
    dtype: object




```python
df_train.number_of_doors.head() # treated as float datatype by the model
```




    2735     2.0
    6720     4.0
    5878     4.0
    11190    3.0
    4554     4.0
    Name: number_of_doors, dtype: float64



**As we checked datatypes, number_of_doors marked as 'float64' instead of Object datatype**<br>
**We have to change the datatype, lets check the categorical columns**


```python
df_train.select_dtypes('object').columns # displays categorical columns in the dataset
```




    Index(['make', 'model', 'engine_fuel_type', 'transmission_type',
           'driven_wheels', 'market_category', 'vehicle_size', 'vehicle_style'],
          dtype='object')




```python
df_train.select_dtypes('object').head() # first 5 rows of categorical variables
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
      <th>make</th>
      <th>model</th>
      <th>engine_fuel_type</th>
      <th>transmission_type</th>
      <th>driven_wheels</th>
      <th>market_category</th>
      <th>vehicle_size</th>
      <th>vehicle_style</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2735</th>
      <td>chevrolet</td>
      <td>cobalt</td>
      <td>regular_unleaded</td>
      <td>manual</td>
      <td>front_wheel_drive</td>
      <td>NaN</td>
      <td>compact</td>
      <td>coupe</td>
    </tr>
    <tr>
      <th>6720</th>
      <td>toyota</td>
      <td>matrix</td>
      <td>regular_unleaded</td>
      <td>automatic</td>
      <td>front_wheel_drive</td>
      <td>hatchback</td>
      <td>compact</td>
      <td>4dr_hatchback</td>
    </tr>
    <tr>
      <th>5878</th>
      <td>subaru</td>
      <td>impreza</td>
      <td>regular_unleaded</td>
      <td>automatic</td>
      <td>all_wheel_drive</td>
      <td>hatchback</td>
      <td>compact</td>
      <td>4dr_hatchback</td>
    </tr>
    <tr>
      <th>11190</th>
      <td>volkswagen</td>
      <td>vanagon</td>
      <td>regular_unleaded</td>
      <td>manual</td>
      <td>rear_wheel_drive</td>
      <td>NaN</td>
      <td>large</td>
      <td>passenger_minivan</td>
    </tr>
    <tr>
      <th>4554</th>
      <td>ford</td>
      <td>f-150</td>
      <td>flex-fuel_(unleaded/e85)</td>
      <td>automatic</td>
      <td>four_wheel_drive</td>
      <td>flex_fuel</td>
      <td>large</td>
      <td>crew_cab_pickup</td>
    </tr>
  </tbody>
</table>
</div>




```python
# How to change number of doors datatype
(df_train.number_of_doors==4).astype(int)
# converts the True/False from condition to integer '0s' and '1s'
```




    2735     0
    6720     1
    5878     1
    11190    0
    4554     1
            ..
    434      0
    1902     0
    9334     1
    5284     1
    2420     1
    Name: number_of_doors, Length: 7150, dtype: int64




```python
# Instead of doing this for v = 2, v = 3, v = 4 and create columns, we can do it one step
'num_doors_%s' %4
```




    'num_doors_4'




```python
for doors in [2,3,4]:
    print('num_doors_%s' %doors)
```

    num_doors_2
    num_doors_3
    num_doors_4



```python
# applying it in our prepare_X function

def prepare_X(df):
    # copying just to make sure adding/modifying new features shouldn't affect original data
    df = df.copy()
    features = base.copy() # Creating Copy of base features

    df['age'] = 2017 - df['year'] # creating new feature in copied dataset
    features.append('age') # adding 'age' feature with existing features in base list

    for doors in [2,3,4]:
        # Creating new columns for each door value 
        # Convert them into binary value wherever condition meets (One-Hot Encoder)
        df['num_doors_%s' %doors] = (df.number_of_doors == doors).astype(int)
        features.append('num_doors_%s' %doors) # Adding new features to existing feature list
        
    df_num = df[features] # selecting features from df
    df_num = df_num.fillna(0) # filling missing values with 0
    X = df_num.values # converting dataframe to NumPy array
    return X
```


```python
X_train = prepare_X(df_train) # preparing X_train dataset after cleaning missing values
w0, w = linear_regression(X_train,y_train) # produces weights of base and parameters

X_val = prepare_X(df_val) # Preparing validation dataset to compare with train dataset
y_pred = w0 + X_val.dot(w) # applying known values of w0, w from train dataset

# Comparing the evaluated validation prediction with existing validation target values
rmse(y_val,y_pred)
```




    0.5157995641502353



- As we checked with previous RMSE value, only slightly decreased <br>
- Let's add top car make as columns and check how the model performs

#### Car Make


```python
list(df.make.value_counts().head().index) # Select the top car brands and do one-hot encoding
```




    ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']




```python
car_make = ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']
# Add this to our prepare_X function
```


```python
# applying it in our prepare_X function

def prepare_X(df):
    # copying just to make sure adding/modifying new features shouldn't affect original data
    df = df.copy()
    features = base.copy() # Creating Copy of base features

    df['age'] = 2017 - df['year'] # creating new feature in copied dataset
    features.append('age') # adding 'age' feature with existing features in base list

    for doors in [2,3,4]:
        # Creating new columns for each door value 
        # Convert them into binary value wherever condition meets (One-Hot Encoder)
        df['num_doors_%s' %doors] = (df.number_of_doors == doors).astype(int)
        features.append('num_doors_%s' %doors) # Adding new features to existing feature list

    for brand in car_make:
        # Creating new columns for each top car brand 
        df['car_%s' %brand] = (df.make == brand).astype(int)
        features.append('car_%s' %brand)
        
    df_num = df[features] # selecting features from df
    df_num = df_num.fillna(0) # filling missing values with 0
    X = df_num.values # converting dataframe to NumPy array
    return X
```


```python
X_train = prepare_X(df_train) # preparing X_train dataset after cleaning missing values
w0, w = linear_regression(X_train,y_train) # produces weights of base and parameters

X_val = prepare_X(df_val) # Preparing validation dataset to compare with train dataset
y_pred = w0 + X_val.dot(w) # applying known values of w0, w from train dataset

# Comparing the evaluated validation prediction with existing validation target values
rmse(y_val,y_pred)
```




    0.5076038849557035



- After appending car make as feature in our training, RMSE value drcreased from 0.517 to 0.507

#### Adding more features

- What happens if we add some more features? Will model reduce error rate? or Will Overfit?


```python
# Select some more feature from categorical datatypes
categorical_columns = [
    'make', 'model', 'engine_fuel_type', 'driven_wheels', 'market_category',
    'vehicle_size', 'vehicle_style']

categorical = {}

for c in categorical_columns:
    # Selecting top most 5 unique values from each column and making them new column in training
    categorical[c] = list(df_train[c].value_counts().head().index)
```


```python
# applying in our prepare_X function
def prepare_X(df):
    # copying just to make sure adding/modifying new features shouldn't affect original data
    df = df.copy()
    features = base.copy() # Creating Copy of base features

    df['age'] = 2017 - df['year'] # creating new feature in copied dataset
    features.append('age') # adding 'age' feature with existing features in base list

    for doors in [2,3,4]:
        # Creating new columns for each door value 
        # Convert them into binary value wherever condition meets (One_Hot Encoder)
        df['num_doors_%s' %doors] = (df.number_of_doors == doors).astype(int)
        features.append('num_doors_%s' %doors) # Adding new features to existing feature list

    for name, values in categorical.items():
        # name - name of the column
        # values - top most value in each column and looped to create as new feature
        for value in values:
            df['%s_%s' % (name, value)] = (df[name] == value).astype(int)
            features.append('%s_%s' % (name, value))

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values

    return X
```


```python
X_train = prepare_X(df_train) # preparing X_train dataset after cleaning missing values
w0, w = linear_regression(X_train,y_train) # produces weights of base and parameters

X_val = prepare_X(df_val) # Preparing validation dataset to compare with train dataset
y_pred = w0 + X_val.dot(w) # applying known values of w0, w from train dataset

# Comparing the evaluated validation prediction with existing validation target values
rmse(y_val,y_pred)
```




    24.780192001181355



- As we checked, RMSE value increased after adding more features. 

### Regularization

The higher value of RMSE maybe due to some values in features are identical to other feature values


```python
# For example
X = [
    [4, 4, 4],
    [3, 5, 5],
    [5, 1, 1],
    [5, 4, 4],
    [7, 5, 5],
    [4, 5, 5.00000001],
] # Column 2 and 3 are actually duplicates

X = np.array(X)
y = [1,2,3,1,2,3]

X, y
```




    (array([[4.        , 4.        , 4.        ],
            [3.        , 5.        , 5.        ],
            [5.        , 1.        , 1.        ],
            [5.        , 4.        , 4.        ],
            [7.        , 5.        , 5.        ],
            [4.        , 5.        , 5.00000001]]),
     [1, 2, 3, 1, 2, 3])




```python
# Calculate normal equation
XTX = X.T.dot(X)
XTX
```




    array([[140.        , 111.        , 111.00000004],
           [111.        , 108.        , 108.00000005],
           [111.00000004, 108.00000005, 108.0000001 ]])




```python
XTX_inv = np.linalg.inv(XTX)
XTX_inv
```




    array([[ 3.86409478e-02, -1.26839821e+05,  1.26839770e+05],
           [-1.26839767e+05,  2.88638033e+14, -2.88638033e+14],
           [ 1.26839727e+05, -2.88638033e+14,  2.88638033e+14]])




```python
XTX_inv.dot(X.T).dot(y)
```




    array([      -0.19390888, -3618543.74936484,  3618546.42894508])



**As we check, w0 is -0.194 and w are above 3.6 * 10^6 which are huge and would impact in RMSE like before <br> To Tackle this problem, we add a small number/value at the diagonals**


```python
# lets get into action how we can do this
XTX = [
    [1,2,2],
    [2,1,1.0001],
    [2,1.0001,1]
]

XTX = np.array(XTX)
XTX
```




    array([[1.    , 2.    , 2.    ],
           [2.    , 1.    , 1.0001],
           [2.    , 1.0001, 1.    ]])




```python
np.linalg.inv(XTX) # 1
```




    array([[   -0.33335556,     0.33333889,     0.33333889],
           [    0.33333889, -5000.08333472,  4999.91666528],
           [    0.33333889,  4999.91666528, -5000.08333472]])



- The column values at 2nd and 3rd column are high
- We will try to add small number (0.01) at the diagonal and check the results


```python
# lets get into action how we can do this
XTX = [
    [1.01,2,2],
    [2,1.01,1.0001],
    [2,1.0001,1.01]
]

XTX = np.array(XTX)
XTX
```




    array([[1.01  , 2.    , 2.    ],
           [2.    , 1.01  , 1.0001],
           [2.    , 1.0001, 1.01  ]])




```python
np.linalg.inv(XTX) # 2
```




    array([[ -0.3367115 ,   0.33501965,   0.33501965],
           [  0.33501965,  50.42045804, -50.58964297],
           [  0.33501965, -50.58964297,  50.42045804]])



- Comparing # 1 and # 2, the values are decreased much


```python
# How to do add the number
XTX = [
    [1,2, 2],
    [2,1,1.0001],
    [2,1.0001,1]
       ]
XTX
```




    [[1, 2, 2], [2, 1, 1.0001], [2, 1.0001, 1]]




```python
XTX = XTX + 0.01 * np.eye(3)
XTX
```




    array([[1.01  , 2.    , 2.    ],
           [2.    , 1.01  , 1.0001],
           [2.    , 1.0001, 1.01  ]])




```python
np.linalg.inv(XTX)
```




    array([[ -0.3367115 ,   0.33501965,   0.33501965],
           [  0.33501965,  50.42045804, -50.58964297],
           [  0.33501965, -50.58964297,  50.42045804]])



***


```python
# Applying the regularization parameter in our linear regression function
def linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0]) # Creating bias term for dataset
    X = np.column_stack([ones, X]) # Adding column wise, bias with dataset

    XTX = X.T.dot(X) # Matrix Multiplication
    XTX = XTX + r * np.eye(XTX.shape[0]) # Adding regularization parameter at the diagonals

    XTX_inv = np.linalg.inv(XTX) # Inverse of XTX
    w_full = XTX_inv.dot(X.T).dot(y) # Normal equation to find the coefficients of bias and weights
    
    return w_full[0], w_full[1:] # Bias term, Weights - w1,..wn
```


```python
X_train = prepare_X(df_train)
w0, w = linear_regression_reg(X_train, y_train, r=0.01)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)
```




    0.4608208286029829



**Our earlier RMSE value is 0.51 and now reduced to 0.46**<br>
**But we have to find the exact 'r' value to find the least RMSE value**

## Tuning the Model


```python
for r_value in [0.0,0.01,0.001,0.0001,0.00001,10]: # Manually given r_values
    # Generate training results for each r_value to find out the optimum r-value
    X_train = prepare_X(df_train)
    w0, w = linear_regression_reg(X_train, y_train, r=r_value)

    X_val = prepare_X(df_val)
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)

    print(r_value, w0, score)
```

    0.0 -1892716650172720.2 24.780192001181355
    0.01 7.1183820236285555 0.4608208286029829
    0.001 7.130829068671119 0.4608158583369783
    0.0001 7.139881370176266 0.46081536403011203
    1e-05 3.6757689730217615 0.46081532315296586
    10 4.729512585698256 0.472609877266825


***

## Using the Model

- Apply everything
- Combine train + validation as train dataset 
- Compare results with test dataset 


```python
# Combine df_train and df_val as single train dataset
df_full_train = pd.concat([df_train,df_val])
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
      <th>make</th>
      <th>model</th>
      <th>year</th>
      <th>engine_fuel_type</th>
      <th>engine_hp</th>
      <th>engine_cylinders</th>
      <th>transmission_type</th>
      <th>driven_wheels</th>
      <th>number_of_doors</th>
      <th>market_category</th>
      <th>vehicle_size</th>
      <th>vehicle_style</th>
      <th>highway_mpg</th>
      <th>city_mpg</th>
      <th>popularity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2735</th>
      <td>chevrolet</td>
      <td>cobalt</td>
      <td>2008</td>
      <td>regular_unleaded</td>
      <td>148.0</td>
      <td>4.0</td>
      <td>manual</td>
      <td>front_wheel_drive</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>compact</td>
      <td>coupe</td>
      <td>33</td>
      <td>24</td>
      <td>1385</td>
    </tr>
    <tr>
      <th>6720</th>
      <td>toyota</td>
      <td>matrix</td>
      <td>2012</td>
      <td>regular_unleaded</td>
      <td>132.0</td>
      <td>4.0</td>
      <td>automatic</td>
      <td>front_wheel_drive</td>
      <td>4.0</td>
      <td>hatchback</td>
      <td>compact</td>
      <td>4dr_hatchback</td>
      <td>32</td>
      <td>25</td>
      <td>2031</td>
    </tr>
    <tr>
      <th>5878</th>
      <td>subaru</td>
      <td>impreza</td>
      <td>2016</td>
      <td>regular_unleaded</td>
      <td>148.0</td>
      <td>4.0</td>
      <td>automatic</td>
      <td>all_wheel_drive</td>
      <td>4.0</td>
      <td>hatchback</td>
      <td>compact</td>
      <td>4dr_hatchback</td>
      <td>37</td>
      <td>28</td>
      <td>640</td>
    </tr>
    <tr>
      <th>11190</th>
      <td>volkswagen</td>
      <td>vanagon</td>
      <td>1991</td>
      <td>regular_unleaded</td>
      <td>90.0</td>
      <td>4.0</td>
      <td>manual</td>
      <td>rear_wheel_drive</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>large</td>
      <td>passenger_minivan</td>
      <td>18</td>
      <td>16</td>
      <td>873</td>
    </tr>
    <tr>
      <th>4554</th>
      <td>ford</td>
      <td>f-150</td>
      <td>2017</td>
      <td>flex-fuel_(unleaded/e85)</td>
      <td>385.0</td>
      <td>8.0</td>
      <td>automatic</td>
      <td>four_wheel_drive</td>
      <td>4.0</td>
      <td>flex_fuel</td>
      <td>large</td>
      <td>crew_cab_pickup</td>
      <td>21</td>
      <td>15</td>
      <td>5657</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Seem the index is shuffled, lets reset the index
df_full_train = df_full_train.reset_index(drop=True) # drop = True will drop existing index
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
      <th>make</th>
      <th>model</th>
      <th>year</th>
      <th>engine_fuel_type</th>
      <th>engine_hp</th>
      <th>engine_cylinders</th>
      <th>transmission_type</th>
      <th>driven_wheels</th>
      <th>number_of_doors</th>
      <th>market_category</th>
      <th>vehicle_size</th>
      <th>vehicle_style</th>
      <th>highway_mpg</th>
      <th>city_mpg</th>
      <th>popularity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>chevrolet</td>
      <td>cobalt</td>
      <td>2008</td>
      <td>regular_unleaded</td>
      <td>148.0</td>
      <td>4.0</td>
      <td>manual</td>
      <td>front_wheel_drive</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>compact</td>
      <td>coupe</td>
      <td>33</td>
      <td>24</td>
      <td>1385</td>
    </tr>
    <tr>
      <th>1</th>
      <td>toyota</td>
      <td>matrix</td>
      <td>2012</td>
      <td>regular_unleaded</td>
      <td>132.0</td>
      <td>4.0</td>
      <td>automatic</td>
      <td>front_wheel_drive</td>
      <td>4.0</td>
      <td>hatchback</td>
      <td>compact</td>
      <td>4dr_hatchback</td>
      <td>32</td>
      <td>25</td>
      <td>2031</td>
    </tr>
    <tr>
      <th>2</th>
      <td>subaru</td>
      <td>impreza</td>
      <td>2016</td>
      <td>regular_unleaded</td>
      <td>148.0</td>
      <td>4.0</td>
      <td>automatic</td>
      <td>all_wheel_drive</td>
      <td>4.0</td>
      <td>hatchback</td>
      <td>compact</td>
      <td>4dr_hatchback</td>
      <td>37</td>
      <td>28</td>
      <td>640</td>
    </tr>
    <tr>
      <th>3</th>
      <td>volkswagen</td>
      <td>vanagon</td>
      <td>1991</td>
      <td>regular_unleaded</td>
      <td>90.0</td>
      <td>4.0</td>
      <td>manual</td>
      <td>rear_wheel_drive</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>large</td>
      <td>passenger_minivan</td>
      <td>18</td>
      <td>16</td>
      <td>873</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ford</td>
      <td>f-150</td>
      <td>2017</td>
      <td>flex-fuel_(unleaded/e85)</td>
      <td>385.0</td>
      <td>8.0</td>
      <td>automatic</td>
      <td>four_wheel_drive</td>
      <td>4.0</td>
      <td>flex_fuel</td>
      <td>large</td>
      <td>crew_cab_pickup</td>
      <td>21</td>
      <td>15</td>
      <td>5657</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_full_train
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
      <th>make</th>
      <th>model</th>
      <th>year</th>
      <th>engine_fuel_type</th>
      <th>engine_hp</th>
      <th>engine_cylinders</th>
      <th>transmission_type</th>
      <th>driven_wheels</th>
      <th>number_of_doors</th>
      <th>market_category</th>
      <th>vehicle_size</th>
      <th>vehicle_style</th>
      <th>highway_mpg</th>
      <th>city_mpg</th>
      <th>popularity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>chevrolet</td>
      <td>cobalt</td>
      <td>2008</td>
      <td>regular_unleaded</td>
      <td>148.0</td>
      <td>4.0</td>
      <td>manual</td>
      <td>front_wheel_drive</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>compact</td>
      <td>coupe</td>
      <td>33</td>
      <td>24</td>
      <td>1385</td>
    </tr>
    <tr>
      <th>1</th>
      <td>toyota</td>
      <td>matrix</td>
      <td>2012</td>
      <td>regular_unleaded</td>
      <td>132.0</td>
      <td>4.0</td>
      <td>automatic</td>
      <td>front_wheel_drive</td>
      <td>4.0</td>
      <td>hatchback</td>
      <td>compact</td>
      <td>4dr_hatchback</td>
      <td>32</td>
      <td>25</td>
      <td>2031</td>
    </tr>
    <tr>
      <th>2</th>
      <td>subaru</td>
      <td>impreza</td>
      <td>2016</td>
      <td>regular_unleaded</td>
      <td>148.0</td>
      <td>4.0</td>
      <td>automatic</td>
      <td>all_wheel_drive</td>
      <td>4.0</td>
      <td>hatchback</td>
      <td>compact</td>
      <td>4dr_hatchback</td>
      <td>37</td>
      <td>28</td>
      <td>640</td>
    </tr>
    <tr>
      <th>3</th>
      <td>volkswagen</td>
      <td>vanagon</td>
      <td>1991</td>
      <td>regular_unleaded</td>
      <td>90.0</td>
      <td>4.0</td>
      <td>manual</td>
      <td>rear_wheel_drive</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>large</td>
      <td>passenger_minivan</td>
      <td>18</td>
      <td>16</td>
      <td>873</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ford</td>
      <td>f-150</td>
      <td>2017</td>
      <td>flex-fuel_(unleaded/e85)</td>
      <td>385.0</td>
      <td>8.0</td>
      <td>automatic</td>
      <td>four_wheel_drive</td>
      <td>4.0</td>
      <td>flex_fuel</td>
      <td>large</td>
      <td>crew_cab_pickup</td>
      <td>21</td>
      <td>15</td>
      <td>5657</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9527</th>
      <td>volvo</td>
      <td>v60</td>
      <td>2015</td>
      <td>regular_unleaded</td>
      <td>240.0</td>
      <td>4.0</td>
      <td>automatic</td>
      <td>front_wheel_drive</td>
      <td>4.0</td>
      <td>luxury</td>
      <td>midsize</td>
      <td>wagon</td>
      <td>37</td>
      <td>25</td>
      <td>870</td>
    </tr>
    <tr>
      <th>9528</th>
      <td>maserati</td>
      <td>granturismo_convertible</td>
      <td>2015</td>
      <td>premium_unleaded_(required)</td>
      <td>444.0</td>
      <td>8.0</td>
      <td>automatic</td>
      <td>rear_wheel_drive</td>
      <td>2.0</td>
      <td>exotic,luxury,high-performance</td>
      <td>midsize</td>
      <td>convertible</td>
      <td>20</td>
      <td>13</td>
      <td>238</td>
    </tr>
    <tr>
      <th>9529</th>
      <td>cadillac</td>
      <td>escalade_hybrid</td>
      <td>2013</td>
      <td>regular_unleaded</td>
      <td>332.0</td>
      <td>8.0</td>
      <td>automatic</td>
      <td>rear_wheel_drive</td>
      <td>4.0</td>
      <td>luxury,hybrid</td>
      <td>large</td>
      <td>4dr_suv</td>
      <td>23</td>
      <td>20</td>
      <td>1624</td>
    </tr>
    <tr>
      <th>9530</th>
      <td>mitsubishi</td>
      <td>lancer</td>
      <td>2016</td>
      <td>regular_unleaded</td>
      <td>148.0</td>
      <td>4.0</td>
      <td>manual</td>
      <td>front_wheel_drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>compact</td>
      <td>sedan</td>
      <td>34</td>
      <td>24</td>
      <td>436</td>
    </tr>
    <tr>
      <th>9531</th>
      <td>kia</td>
      <td>sorento</td>
      <td>2015</td>
      <td>regular_unleaded</td>
      <td>290.0</td>
      <td>6.0</td>
      <td>automatic</td>
      <td>front_wheel_drive</td>
      <td>4.0</td>
      <td>crossover</td>
      <td>midsize</td>
      <td>4dr_suv</td>
      <td>25</td>
      <td>18</td>
      <td>1720</td>
    </tr>
  </tbody>
</table>
<p>9532 rows × 15 columns</p>
</div>




```python
# From earlier written function to do feature engg 
def prepare_X(df):
    # copying just to make sure adding/modifying new features shouldn't affect original data
    df = df.copy()
    features = base.copy() # Creating Copy of base features

    df['age'] = 2017 - df['year'] # creating new feature in copied dataset
    features.append('age') # adding 'age' feature with existing features in base list

    for doors in [2,3,4]:
        # Creating new columns for each door value 
        # Convert them into binary value wherever condition meets (One_Hot Encoder)
        df['num_doors_%s' %doors] = (df.number_of_doors == doors).astype(int)
        features.append('num_doors_%s' %doors) # Adding new features to existing feature list

    for name, values in categorical.items():
        # name - name of the column
        # values - top most value in each column and looped to create as new feature
        for value in values:
            df['%s_%s' % (name, value)] = (df[name] == value).astype(int)
            features.append('%s_%s' % (name, value))

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values

    return X
```


```python
# prepare and clean full dataset
X_full_train = prepare_X(df_full_train)
X_full_train
```




    array([[148.,   4.,  33., ...,   1.,   0.,   0.],
           [132.,   4.,  32., ...,   0.,   0.,   1.],
           [148.,   4.,  37., ...,   0.,   0.,   1.],
           ...,
           [332.,   8.,  23., ...,   0.,   0.,   0.],
           [148.,   4.,  34., ...,   0.,   0.,   0.],
           [290.,   6.,  25., ...,   0.,   0.,   0.]])




```python
# Combine y values of train and validation dataset together
y_full_train = np.concatenate([y_train,y_val])
y_full_train
```




    array([ 9.57574708,  9.887663  ,  9.89323518, ..., 11.21756062,
            9.77542688, 10.1924563 ])




```python
# train the x_full_train and y_full_train dataset to find the coefficients
w0, w = linear_regression_reg(X_full_train, y_full_train, r=0.001)
```


```python
# Apply w0, w to find the prediction values
X_test = prepare_X(df_test) # feature engg on features
y_pred = w0 + X_test.dot(w) # prediction on X_test values
score = rmse(y_test, y_pred) # comparing existing y values with predicted values
score
```




    0.46007539687771004



## Testing Out the Model

- Lets apply the model on an unseen data and check its performance


```python
df_test.iloc[5]
```




    make                                           audi
    model                                            a3
    year                                           2015
    engine_fuel_type     premium_unleaded_(recommended)
    engine_hp                                     220.0
    engine_cylinders                                4.0
    transmission_type                  automated_manual
    driven_wheels                       all_wheel_drive
    number_of_doors                                 2.0
    market_category                              luxury
    vehicle_size                                compact
    vehicle_style                           convertible
    highway_mpg                                      32
    city_mpg                                         23
    popularity                                     3105
    Name: 1027, dtype: object




```python
car = df_test.iloc[5].to_dict()
car
```




    {'make': 'audi',
     'model': 'a3',
     'year': 2015,
     'engine_fuel_type': 'premium_unleaded_(recommended)',
     'engine_hp': 220.0,
     'engine_cylinders': 4.0,
     'transmission_type': 'automated_manual',
     'driven_wheels': 'all_wheel_drive',
     'number_of_doors': 2.0,
     'market_category': 'luxury',
     'vehicle_size': 'compact',
     'vehicle_style': 'convertible',
     'highway_mpg': 32,
     'city_mpg': 23,
     'popularity': 3105}




```python
# To make this test data to give results, 
# we have to modify the features as like in train data by prepare_X function
# To do that, we have convert that into dataframe
# remember prepare_X function accepts dataframe only

df_test_car = pd.DataFrame([car])
df_test_car
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
      <th>make</th>
      <th>model</th>
      <th>year</th>
      <th>engine_fuel_type</th>
      <th>engine_hp</th>
      <th>engine_cylinders</th>
      <th>transmission_type</th>
      <th>driven_wheels</th>
      <th>number_of_doors</th>
      <th>market_category</th>
      <th>vehicle_size</th>
      <th>vehicle_style</th>
      <th>highway_mpg</th>
      <th>city_mpg</th>
      <th>popularity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>audi</td>
      <td>a3</td>
      <td>2015</td>
      <td>premium_unleaded_(recommended)</td>
      <td>220.0</td>
      <td>4.0</td>
      <td>automated_manual</td>
      <td>all_wheel_drive</td>
      <td>2.0</td>
      <td>luxury</td>
      <td>compact</td>
      <td>convertible</td>
      <td>32</td>
      <td>23</td>
      <td>3105</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Now apply on prepare_X function to create features in age, no.doors and for categorical variables
X_test_car = prepare_X(df_test_car)
```


```python
# Lets check our model prediction on our test car details
y_pred = w0 + X_test_car.dot(w) # We already know w0, w
y_pred = y_pred[0]
y_pred
```




    10.474552314979627




```python
# Our model predicted in logarithmic values, convert them to check values in actual MSRP
np.expm1(y_pred)
```




    35402.01676990125




```python
# Lets check with our actual y value
np.expm1(y_test[5])
```




    41149.999999999985



By Comparing, we came to know that we're around $ 5K lower than actual which is actually good model as we did only few feature engg and trained with few variables.


