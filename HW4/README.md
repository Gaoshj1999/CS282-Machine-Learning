# README

# Introduction

The data consist of local crime statistics for $1994$ US communities. 

The response $y$ is the **crime rate**. The name of the response variable is `ViolentCrimesPerPop`. 

There are $95$ features of the input $\mathbf{x}$ corresponding to the crime rate $y$ ($\mathbf{x}\in\mathbb{R}^{95}$). These features include possibly relevant variables such as the size of the police force or the percentage of children that graduate high school. 

Our data is composed of two parts:

- Training dataset, `crime-train.txt`, which has $1595$ samples, with the first column holding the label $y$, the rest columns holding the input $\mathbf{x}$ .
- Test dataset, `crime-test.txt`, which has $399$ samples with the same configuration as the training dataset.

The features have been standardized to have mean $0$ and variance $1$.

## Usage

### Python

You can read them in the files with in Python:

```python
import pandas as pd
df_train = pd.read_table("crime-train.txt")
df_test = pd.read_table("crime-test.txt")
```

This stores the data as Pandas DataFrame objects. 

DataFrames are similar to Numpy arrays but more flexible; unlike Numpy arrays, they store row and column indices along with the values of the data. Each column of a DataFrame can also, in principle, store data of a different type. For this assignment, however, all data are floats. Here are a few commands that will get you working with Pandas for this assignment:

```python
# Print the first few lines of DataFrame df.
df.head()
# Get the row indices for df.
df.index   
# Get the column indices.
df.columns   
# Return the column named ``foo'.
df[''foo'']    
# Return all columns except ``foo''.
df.drop(''foo'', axis = 1) 
# Return the values as a Numpy array.
df.values        
# Grab column foo and convert to Numpy array.
df[''foo'].values     
# Use numerical indices (like Numpy) to get 3 rows and cols.
df.iloc[:3,:3]              
```

### Matlab

To import the dataset, you can use the `readtable` function, which returns a `table` in matlab.

```matlab
train_data = readtable('crime-train.txt')
```

The table is something like `struct` in C language, which you can visit the data by it's name:

```matlab
train_label = train_data.ViolentCrimesPerPop
```

For more information, please refer to [Access Data in Tables](https://www.mathworks.com/help/matlab/matlab_prog/access-data-in-a-table.html).



## Contact

If you have any further problems, send email to TA `maosong@shanghaitech.edu.cn`.