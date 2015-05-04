# Exploring Kaggle Titanic Data Using Pandas in Python

This is the first workshop of the Python Data Science Basics Series. It's designed for Python/pandas beginners to get started right away. Therefore, we don't necessarily introduce all the concepts in this workshop, but important concepts are included and we will elaborate on them later on. This workshop is inspired by a number of resources:
* [Python for Data Analysis](http://shop.oreilly.com/product/0636920023784.do) This is written by the creator of pandas Wes McKinney
* [The Official Pandas Documentation](http://pandas.pydata.org/pandas-docs/stable/index.html)
* [Data Carpentry Python Notes](https://github.com/datacarpentry/python-ecology)
* [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/details/getting-started-with-python-ii)
* [Pandas Cookbook By Julia Evans](https://github.com/jvns/pandas-cookbook#pandas-cookbook)
* [Pandas Lessons From Hernan Rojas](https://bitbucket.org/hrojas/learn-pandas)

We recommend installing [Python 2.7 from Anaconda](http://continuum.io/downloads#all), which comes with all the libraries and tools that we use in this lesson. We use the csv file 'train.csv' from [Titanic data](https://www.kaggle.com/c/titanic/data) for demos in this lesson. The lesson notes assumes that you saved 'train.csv' in the directory called 'data'.

Let's start by opening a terminal window (windows users can open a Python command console in the Anaconda program). First make sure that you are in the parent directory of 'data' (one level above). Type 'ls' (or 'dir' for windows users) and you should see this:

	data

Okay! Now let's dive into Python by typing 'ipython'. IPython is an enhanced envrionment for writing Python code. If you like just using Python, you can type 'python' instead (but you will miss some of the cool features from IPython, like tab completion.). If you use IPython, you should see something similar to this:

	Python 2.7.9 |Anaconda 2.1.0 (x86_64)| (default, Dec 15 2014, 10:37:34) 
	Type "copyright", "credits" or "license" for more information.

	IPython 3.1.0 -- An enhanced Interactive Python.
	Anaconda is brought to you by Continuum Analytics.
	Please check out: http://continuum.io/thanks and https://binstar.org
	?         -> Introduction and overview of IPython's features.
	%quickref -> Quick reference.
	help      -> Python's own help system.
	object?   -> Details about 'object', use 'object??' for extra details.

	In [1]: 


Before doing anything with the dataset, it's always a good idea to take a close look at the [data description](https://www.kaggle.com/c/titanic/data) from the source of the dataset. Usually it gives a comprehensive background of what each variable means.

## Loading libraries and looking at data

There are different ways to load a library.

	import pandas
	import pandas as pd
	from pandas import *

We recommend 'import pandas as pd', as it retains the specificity of the library without too much typing. Plus, it's the convention that the Python data science community uses.

If you want to see the version of pandas you are using, you can type:
	
	pd.__version__

The dot after 'pd' is a way to get the object, or method (tools).How do we know what to type after 'pd.'? We can tab complete (press on the 'tab' key) after typing the following command to get all the methods available for us.
	
	train = pd.

To load this data file, we use the '.read_csv' method:
	
	train = pd.read_csv('./data/train.csv')

What does our data look like? Let's take a look at the first 5 rows:
	
	train.head()

Or the last 5 rows:
	
	train.tail()

If you just type 'train', you will have all the results flooding your screen...
	
	train
	
We can use '.head()' in diffent ways
	
	train.head(3)
	train.head(n=3)

but how do we know what parameters to use without googling? We can use ? for help like this:
	
	train.head?

### to look at the data
	
	train.info()

Train is a type of data called 'dataframe' in pandas. If you are an R user, you know what it means ;) '.info()'' gives you a summary of the data structure of the dataframe. If you only want to look at the data types of its attributes, you can try:
	
	train.dtypes

We will go back to data types later.

## Indexing & Slicing in Python

### select all data from a column named 'age'
	
	train['Age']
or
	
	train.Age

Both commands will give you the same results, although 'train['age']' is easier to spot in many lines of codes that use various dot methods. If you want to know how long each process takes, you can type:
	
	%time train['Age']

create a new object that contains the age column
	
	train_age = train['Age']

select more than one columns
	
	train[['Age', 'Pclass']].head()

select the 1st, 2nd, and 3rd row from train_age
	
	train_age[0:3]

Python starts its index from 0, and the value after ':' is where it ends, but not included. So the first element in train_age is:
	
	train_age[0]

select the first 5 rows from train_age
	
	train_age[:5]

## Challenge: select the 2nd to 14th rows from column pclass
	
	train['Pclass'][1:15]

how do we know if that's correct? we can look at the index numbers before the values

select the last row from train_age
	
	train_age[-1:]

or you can select the 3rd last row to the 2nd last row
	
	train_age[-3:-1]

## editing the dataframe
Set first 3 row values to zero
first we make a copy of the train dataframe:
	
	train1 = train.copy()
	train1['Age'][0:3] = 0
now we can check and compare:
	
	train1['Age'].head()
	train['Age'].head()

### slicing subsets of rows and columns
	
	train[['Age','Pclass']][3:5]

or use index locations
	
	train.iloc[3:5,[1,4]]
	train.iloc[[1,5],[1,4]] how to show 2,3, then 10:15

## Challenge: what does this one show?
	
	train.iloc[[0,4,10],:]

## Descriptive statistics

### counting and summarizing a data series
counting the frequencies of each unique element in the data series
	
	age_counts = train_age.value_counts()
look at the first 10 records of the count results
	
	age_counts.head(10)

The results are in descending order by default. In this special case, the age_counts data series has integer values. We can  show all the records all the way to a specific value (say age 10 in this case) using

	age_counts[:10]

count the total and skip all the missing values:
	
	train_age.sum(skipna = True)
wondering what happens if you don't skip missing values?
	
	train_age.sum(skipna = False)
it won't calculate the sum because it can't do sum with unknown values! it's a safe way to take the sum and make sure that you are not missing anything.

we can also look at the oldest
	
	train_age.max()

the youngest
	
	train_age.min()
	
the average
	
	train_age.mean()

and the median
	
	train_age.median()

Or we can use a default method to look at the summary statistics of age all together:
	
	train_age.describe()

describing the whole dataframe
	
	train.describe()

wait, where are the string variables? let's take a look at the ticket column:
	
	train['Ticket'].describe()

it turns out that '.describe' displays numeric variables if the dataframe has a mix of different types of varibles. To specify each type, we can use:
	
	train.describe(include =['object'])
	train.describe(include =['number'])

and to look at the whole thing:
	
	train.describe(include ='all')

## Subsetting Data Using Criteria

### subsetting & filtering using Boolean values
pandas does vectorized calculation. This compares every value in train_age with 60, and generates a series of True or False
	
	train_age > 60

Applying the series the same way you select a column to select all the records in the age column at age above 60
	
	train_age[train_age > 60].head()

we can also only select all the 60 yrs olds in the age column
	
	train_age[train.Age == 60]

it's boring to look at only ages, so let's select the whole dataframe using the same criteria, again the first 5 rows
	
	train[train_age == 60].head()

We can 
	
	train[train > 70][:10]
	train.where(train > 70)[:10]

select all the people at age not equal to 60
	
	train[train_age != 60]
or reverse that
	
	train[~(train_age != 60)]

### using multiple criteria
age between 60 and 75
	
	train[(train_age > 60) & (train_age < 75)]

age above 75 or below 2
	
	train[(train_age > 75) | (train_age < 2)]

### Selecting Challenge: 

Select all the people that are:

1. in the 3rd class
2. under age 10
3. female 

Anwser: 

	train[(train['Age'] < 10) & (train['pclass'] == 3) & (train['sex'] == "female")]

A different way to match multiple criteria using 'in'
	
	train[train_age.isin([20,30])]
and 'not in' using the '~' in from of the criteria
	
	train[~train_age.isin([20,30])]

It works on strings too:
	
	train[train['Ticket'].isin(['113803'])]

To select all tickets that match 'PC 17599':
	
	train['Ticket'][train['Ticket'].str.match('PC 17599', as_indexer=True)]
or to get all the records with the same criteria:
	
	train[train['Ticket'].str.match('PC 17599', as_indexer=True)]

We can use 'contains' to find names that has 'Lily' in them
	
	train['Name'][train['Name'].str.contains('Lily')]
	
'startswith' and 'endswith'
	
	train['Name'][train['Name'].str.endswith('Peel)')]
	train['Name'][train['Name'].str.startswith('Futrelle')]

## sorting
sort the age data series in ascending order
	
	train_age.order()

reverse the order
	
	train_age.order(ascending = False)

the first 3 smallest numbers in the age data series
	
	train_age.nsmallest(3)

the first 3 largest numbers in the age data series
	
	train_age.nlargest(3)

### sort by multi-indexing
sort the dataframe by age
	
	train.sort_index(by = ('Age'))

sort the dataframe first by age, then pclass
	
	train.sort_index(by = (['Age','Pclass']))[:10]

sort the dataframe first by class, then age
	
	train.sort_index(by = (['Pclass','Age']))[:10]

## missing values
we can check missing values using the following methods
	
	isnull()
	notnull()

The first 5 missing values in the age dataseries
	
	train_age[train_age.isnull()].head()

The first 5 records that have missing age values
	
	train[train_age.isnull()].head()

You can replace missing values with anything using '.fillna()', but we don't recommend it! Missing values are better left as is, because programming languages usually treat them differently from everything else. If you replace them with any value, they may affect your calculations later on.
If you really have to do it, at least make a copy of the dataframe first:
	
	train2 = train.copy()
then replace all missing age values with 0
	
	train2['Age'] = train2['Age'].fillna('0')

We can also replace missing age values with strings
	
	train3 = train.copy()
	train3['Age'] = train3['Age'].fillna('wow')

Sometimes we may want to remove records that have missing values.We can use '.dropna()' to do that. 
Caution! Removing records with missing values will have serious consequences, so think about it twice before doing it!
	
	train4 = train.copy()
	train4 = train4.dropna()

## adding columns
We can add new columns to the dataframe by directly assigning values to the new column:
	
	train['Gender'] = 4

We can also add new columns using calculations:
	
	train['Fare_to_age'] = train['Fare'] / train['Age']

We can also create columns with missing values:
	
	import numpy as np
	train['Missing'] = np.nan

we can replace 0 with missing values too:
	
	train2['Age'] = train2['Age'].replace(0, np.nan)

## Applying elementwise Python functions in a data series
Adding a column called 'Gender' that is filled by the first letter from the values in 'Sex' column
	
	train['Gender'] = train['Sex'].map(lambda x: x[0].upper())

If we want to apply a function in a dataframe, we use 

	.applymap
This function turns all the values into a string first, then replace its value with its string length
	
	f = lambda x: len(str(x))

Then we apply the function to every value in the dataframe!
	
	train.applymap(f)

If we don't assign the new values to the dataframe, it won't change the original:
	
	train.head()

But if we do this, we replace the original dataframe with new values, and there is no way back unless we re-read the data file again:
	
	train = train.head()

We can write the new dataframe out to a csv file
	
	train.to_csv('train_crazy.csv')
