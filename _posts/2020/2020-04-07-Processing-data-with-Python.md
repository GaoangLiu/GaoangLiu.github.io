---
layout: post
title: Processing data with Python
date: 2020-04-07
tags: python numpy pandas preprocessing
categories: python
author: GaoangLau
---
* content
{:toc}


# Scaler 
## sklearn api

```python



from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
```
* `StandardScaler`, data will transformed into a distribution with a mean value 0 and standard deviation of 1.
  * Calculation:
  * Mean $$\mu = \frac{1}{N} \sum_{i=1}^N(x_i)$$
  * Standard deviation $$\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^N(x_i - \mu)^2}$$
  * Standardization $$z = \frac{x - \mu}{\sigma}$$

* `MinMaxScaler`, $$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

* `RobustScaler`, uses a similar method to the `MinMaxScaler` but uses the interquartile range instead, rather than the min-max, so that it is robust to outliers, `X = (X - Q_1(X)) / (Q_3(X) - Q_1(X))`, where `Q_1(X) = np.quantile(X, 0.25, axis=0), Q_3(X) = np.quantile(X, 0.75, axis=0)`, 
  * By default, `sklearn.preprocessing.RobustScaler` will set `with_centering=True`, which means `np.median(X)` will replace `Q_1(X)` in the scaling formula, i.e., `X = (X - np.median(X)) / (Q_3(X) - Q_1(X))`

--- 

## Numpy 

[Official site](https://numpy.org/)

NumPy’s array class is called `ndarray`, this is not the same thing as the Python Library `array.array` 

*  create sequences of numbers

```python 
np.arange(10, 30, 5) #array([10, 15, 20, 25])
```




# Pandas

Creating a [`Series`](https://pandas.pydata.org/docs/reference/api/pandas.Series.html#pandas.Series) by passing a list of values

```pyth
s = pd.Series([1, 2, 3, np.nan, 9])
```

* sorting by an axis: `df.sort_index(axis=1, ascending=False)`
* sorting by values: `df.sort_values(by="Age")`
* `for i, row in df.iterrows()`is a generator which yields both index and row 



## Columns

1. Take column-slices `df_slice = df[['c1', 'c2', 'c19']]`
2. Renaming columns: `df.columns = ['newc1', 'newc2', 'newc..']`
3. Convert one column to list `df.age.to_list()`
4. Add new columns : `df.assign(col_extra_1 = [...], col_extra_2 = [...])`



## Manupilate Excel

Use `pd.read_excel('data.xlsx', sheet_name="salary")` to read an excel file, and `pd.to_excel('data.xlsx', 'Sheet1', index_col=None, na_values=['NA'])` to write to an excel. 

### How to get the list of sheets ?

Two ways to accomplish this: 

1. `df = pd.read_excel('d.xlsx', None)` and then run `df.keys()`
2. `xl = pd.ExcelFile('d.xlsx'); xl.sheet_names`

If sheetnames is the only thing you care about, the second way is more efficient while the first one unnecessarily parses every sheet as a DataFrame. 

The [ExcelFile](https://pandas.pydata.org/pandas-docs/dev/user_guide/io.html)  class is designed to facilitate working multiple sheets from the same file. With ExcelFile, the file is read to memory only once. 

The primary use-case for an `ExcelFile` is parsing multiple sheets with different parameters:

```python
df = {}
with pd.ExcelFile('d.xlsx') as xls:
  df['sheet_1'] = pd.read_excel(xls, 'Sheet1', index_col=None, na_values=['NA'])
  df['sheet_2'] = pd.read_excel(xls, 'Sheet2', index_col=1)
```

If the parsing parameters are used for all sheets, we can compress the above three lines into one:

```python
df = pd.read_excel('d.xlsx', ['Sheet1', 'Sheet2'], index_col=None, na_values=['NA'])
```



## Refer 

[pandas official](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html)

