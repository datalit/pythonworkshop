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
	In [1]: import pandas as pd

There are alternative ways to load a library.

	import pandas
	from pandas import *

Try to avoid "from pandas import *"! We recommend 'import pandas as pd', as it retains the specificity of the library without too much typing. Plus, it's the convention that the Python data science community uses.

If you want to see the version of pandas you are using, you can type:
	
	In [2]: pd.__version__
	Out[2]: '0.16.2'

The dot after 'pd' is a way to get the object, or method (tools). How do we know what to type after 'pd.'? We can tab complete (press on the 'tab' key) after typing the following command to get all the methods available for us.
	
	In [3]: train = pd.

To load this data file, we use the '.read_csv' method:
	
	In [3]: train = pd.read_csv('./data/train.csv')

The '=' does not compare whether left is equal to right, but rather it is assigning what is on the right to the variable "train" on the left. We will use it all the time to create new variables.

What does our data look like? Let's take a look at the first 5 rows:
	
	In [4]: train.head()
	Out[4]: 
	   PassengerId  Survived  Pclass  \
	0            1         0       3   
	1            2         1       1   
	2            3         1       3   
	3            4         1       1   
	4            5         0       3   
	
	                                                Name     Sex  Age  SibSp  \
	0                            Braund, Mr. Owen Harris    male   22      1   
	1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female   38      1   
	2                             Heikkinen, Miss. Laina  female   26      0   
	3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female   35      1   
	4                           Allen, Mr. William Henry    male   35      0   
	
	   Parch            Ticket     Fare Cabin Embarked  
	0      0         A/5 21171   7.2500   NaN        S  
	1      0          PC 17599  71.2833   C85        C  
	2      0  STON/O2. 3101282   7.9250   NaN        S  
	3      0            113803  53.1000  C123        S  
	4      0            373450   8.0500   NaN        S  

Or the last 5 rows:
	
	In [5]: train.tail()
	Out[5]: 
	     PassengerId  Survived  Pclass                                      Name  \
	886          887         0       2                     Montvila, Rev. Juozas   
	887          888         1       1              Graham, Miss. Margaret Edith   
	888          889         0       3  Johnston, Miss. Catherine Helen "Carrie"   
	889          890         1       1                     Behr, Mr. Karl Howell   
	890          891         0       3                       Dooley, Mr. Patrick   
	
	        Sex  Age  SibSp  Parch      Ticket   Fare Cabin Embarked  
	886    male   27      0      0      211536  13.00   NaN        S  
	887  female   19      0      0      112053  30.00   B42        S  
	888  female  NaN      1      2  W./C. 6607  23.45   NaN        S  
	889    male   26      0      0      111369  30.00  C148        C  
	890    male   32      0      0      370376   7.75   NaN        Q  

If you just type 'train', you will have all the results flooding your screen...
	
	In [6]: train
	Out[6]: 
	     PassengerId  Survived  Pclass  \
	0              1         0       3   
	1              2         1       1   
	2              3         1       3   
	3              4         1       1   
	4              5         0       3   
	5              6         0       3   
	6              7         0       1   
	7              8         0       3   
	8              9         1       3   
	9             10         1       2   
	..     ...               ...       ...          ...      ...  
	861      0             28134   11.5000          NaN        S  
	862      0             17466   25.9292          D17        S  
	863      2          CA. 2343   69.5500          NaN        S  
	864      0            233866   13.0000          NaN        S  
	865      0            236852   13.0000          NaN        S  
	866      0     SC/PARIS 2149   13.8583          NaN        C  
	867      0          PC 17590   50.4958          A24        S  
	868      0            345777    9.5000          NaN        S  
	869      1            347742   11.1333          NaN        S  
	870      0            349248    7.8958          NaN        S  
	871      1             11751   52.5542          D35        S  
	872      0               695    5.0000  B51 B53 B55        S  
	873      0            345765    9.0000          NaN        S  
	874      0         P/PP 3381   24.0000          NaN        C  
	875      0              2667    7.2250          NaN        C  
	876      0              7534    9.8458          NaN        S  
	877      0            349212    7.8958          NaN        S  
	878      0            349217    7.8958          NaN        S  
	879      1             11767   83.1583          C50        C  
	880      1            230433   26.0000          NaN        S  
	881      0            349257    7.8958          NaN        S  
	882      0              7552   10.5167          NaN        S  
	883      0  C.A./SOTON 34068   10.5000          NaN        S  
	884      0   SOTON/OQ 392076    7.0500          NaN        S  
	885      5            382652   29.1250          NaN        Q  
	886      0            211536   13.0000          NaN        S  
	887      0            112053   30.0000          B42        S  
	888      2        W./C. 6607   23.4500          NaN        S  
	889      0            111369   30.0000         C148        C  
	890      0            370376    7.7500          NaN        Q  
	
	[891 rows x 12 columns]

	
We can use '.head()' in diffent ways
	
	In [7]: train.head(3)
	Out[7]: 
	   PassengerId  Survived  Pclass  \
	0            1         0       3   
	1            2         1       1   
	2            3         1       3   
	
	                                                Name     Sex  Age  SibSp  \
	0                            Braund, Mr. Owen Harris    male   22      1   
	1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female   38      1   
	2                             Heikkinen, Miss. Laina  female   26      0   
	
	   Parch            Ticket     Fare Cabin Embarked  
	0      0         A/5 21171   7.2500   NaN        S  
	1      0          PC 17599  71.2833   C85        C  
	2      0  STON/O2. 3101282   7.9250   NaN        S  

	In [8]: train.head(n=3)
	Out[8]: 
	   PassengerId  Survived  Pclass  \
	0            1         0       3   
	1            2         1       1   
	2            3         1       3   
	
	                                                Name     Sex  Age  SibSp  \
	0                            Braund, Mr. Owen Harris    male   22      1   
	1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female   38      1   
	2                             Heikkinen, Miss. Laina  female   26      0   
	
	   Parch            Ticket     Fare Cabin Embarked  
	0      0         A/5 21171   7.2500   NaN        S  
	1      0          PC 17599  71.2833   C85        C  
	2      0  STON/O2. 3101282   7.9250   NaN        S  

but how do we know what parameters to use without googling? We can use ? for help like this:
	
	In [9]: train.head?
	Signature: train.head(n=5)
	Docstring: Returns first n rows
	File:      ~/anaconda/lib/python2.7/site-packages/pandas/core/generic.py
	Type:      instancemethod

or use help(). Remember to press "q" to quit.

	In [10]: help(train.head)
	
### to look at the data
	
	In [11]: train.info()
	<class 'pandas.core.frame.DataFrame'>
	Int64Index: 891 entries, 0 to 890
	Data columns (total 12 columns):
	PassengerId    891 non-null int64
	Survived       891 non-null int64
	Pclass         891 non-null int64
	Name           891 non-null object
	Sex            891 non-null object
	Age            714 non-null float64
	SibSp          891 non-null int64
	Parch          891 non-null int64
	Ticket         891 non-null object
	Fare           891 non-null float64
	Cabin          204 non-null object
	Embarked       889 non-null object
	dtypes: float64(2), int64(5), object(5)
	memory usage: 90.5+ KB

Train is a type of data called 'dataframe' in pandas. If you are an R user, you know what it means ;) '.info()'' gives you a summary of the data structure of the dataframe. If you only want to look at the data types of its attributes, you can try:
	
	In [12]: train.dtypes
	Out[12]: 
	PassengerId      int64
	Survived         int64
	Pclass           int64
	Name            object
	Sex             object
	Age            float64
	SibSp            int64
	Parch            int64
	Ticket          object
	Fare           float64
	Cabin           object
	Embarked        object
	dtype: object

We will go back to data types later.

## Indexing & Slicing in Python

### select all data from a column named 'age'
	
	train['Age']
or
	
	train.Age

	In [13]: train['Age']
	Out[13]: 
	0      22
	1      38
	2      26
	3      35
	4      35
	       ..
	881    33
	882    22
	883    28
	884    25
	885    39
	886    27
	887    19
	888   NaN
	889    26
	890    32
	Name: Age, dtype: float64

Both commands will give you the same results, although 'train['age']' is easier to spot in many lines of codes that use various dot methods. If you want to know how long each process takes, you can type:
	
	In [14]: %time train['Age']
	CPU times: user 44 µs, sys: 1e+03 ns, total: 45 µs
	Wall time: 80.1 µs
	Out[14]: 
	0      22
	1      38
	2      26
	3      35
	4      35
	       ..
	884    25
	885    39
	886    27
	887    19
	888   NaN
	889    26
	890    32
	Name: Age, dtype: float64
create a new object that contains the age column
	
	In [15]: train_age = train['Age']

select more than one columns
	
	In [16]: train[['Age', 'Pclass']].head()
	Out[16]: 
	   Age  Pclass
	0   22       3
	1   38       1
	2   26       3
	3   35       1
	4   35       3

select the 1st, 2nd, and 3rd row from train_age
	
	In [17]: train_age[0:3]
	Out[17]: 
	0    22
	1    38
	2    26
	Name: Age, dtype: float64

Python starts its index from 0, and the value after ':' is where it ends, but not included. So the first element in train_age is:
	
	In [18]: train_age[0]
	Out[18]: 22.0

select the first 5 rows from train_age
	
	In [19]: train_age[:5]
	Out[19]: 
	0    22
	1    38
	2    26
	3    35
	4    35
	Name: Age, dtype: float64

## Challenge: select the 2nd to 14th rows from column pclass
	
	In [20]: train['Pclass'][1:15]
	Out[20]: 
	1     1
	2     3
	3     1
	4     3
	5     3
	6     1
	7     3
	8     3
	9     2
	10    3
	11    1
	12    3
	13    3
	14    3
	Name: Pclass, dtype: int64

how do we know if that's correct? we can look at the index numbers before the values

select the last row from train_age
	
	In [21]: train_age[-1:]
	Out[21]: 
	890    32
	Name: Age, dtype: float64

or you can select the 3rd last row to the 2nd last row
	
	In [22]: train_age[-3:-1]
	Out[22]: 
	888   NaN
	889    26
	Name: Age, dtype: float64

## editing the dataframe
Set first 3 row values to zero
first we make a copy of the train dataframe:
	
	In [23]: train1 = train.copy()
	In [24]: train1['Age'][0:3] = 0
	/Users/xufei/anaconda/bin/ipython:1: SettingWithCopyWarning: 
	A value is trying to be set on a copy of a slice from a DataFrame

	See the the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
	  #!/bin/bash /Users/xufei/anaconda/bin/python.app

Notice the warning message. It was not an error, but it's suggesting that our slicing method is not the best. For detailed explanations please refer to the url above. We will show you different ways of slicing below.

now we can check and compare:
	
	In [25]: train1['Age'].head()
	Out[25]: 
	0     0
	1     0
	2     0
	3    35
	4    35
	Name: Age, dtype: float64

### slicing subsets of rows and columns
	
	In [26]: train[['Age','Pclass']][3:5]
	Out[26]: 
	   Age  Pclass
	3   35       1
	4   35       3

We can use .loc which is a better and more efficient method than chaining labels.

	In [27]: train.loc[:,('Age','Pclass')][3:5]
	Out[27]: 
	   Age  Pclass
	3   35       1
	4   35       3

or we can use location-based indexer .iloc
	
	In [28]: train.iloc[3:5,[1,4]]
	Out[28]: 
	   Survived     Sex
	3         1  female
	4         0    male


	In [29]: train.iloc[[1,5],[1,4]]
	Out[29]: 
	   Survived     Sex
	1         1  female
	5         0    male 

## Challenge: what does this one show?
	
	In [30]: train.iloc[[0,4,10],:]
	Out[30]: 
	    PassengerId  Survived  Pclass                             Name     Sex  \
	0             1         0       3          Braund, Mr. Owen Harris    male   
	4             5         0       3         Allen, Mr. William Henry    male   
	10           11         1       3  Sandstrom, Miss. Marguerite Rut  female   

	    Age  SibSp  Parch     Ticket   Fare Cabin Embarked  
	0    22      1      0  A/5 21171   7.25   NaN        S  
	4    35      0      0     373450   8.05   NaN        S  
	10    4      1      1    PP 9549  16.70    G6        S  

if you want to jump select non-consecutive and consecutive rows or columns, you can use iloc combined with the RClass from the numpy module. first we need to import numpy:
	import numpy as np

then if we want to take the 1st, 3rd, and 5-10th rows from dataframe train, we can use np.r_[0,2,4:10] to concatnate the desired row indices:
	In [30]: train.iloc[np.r_[0,2,4:10], :]
	Out[30]: 
	   PassengerId  Survived  Pclass  \
	0            1         0       3   
	2            3         1       3   
	4            5         0       3   
	5            6         0       3   
	6            7         0       1   
	7            8         0       3   
	8            9         1       3   
	9           10         1       2   
	
	                                                Name     Sex  Age  SibSp  \
	0                            Braund, Mr. Owen Harris    male   22      1   
	2                             Heikkinen, Miss. Laina  female   26      0   
	4                           Allen, Mr. William Henry    male   35      0   
	5                                   Moran, Mr. James    male  NaN      0   
	6                            McCarthy, Mr. Timothy J    male   54      0   
	7                     Palsson, Master. Gosta Leonard    male    2      3   
	8  Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)  female   27      0   
	9                Nasser, Mrs. Nicholas (Adele Achem)  female   14      1   
	
	   Parch            Ticket     Fare Cabin Embarked  
	0      0         A/5 21171   7.2500   NaN        S  
	2      0  STON/O2. 3101282   7.9250   NaN        S  
	4      0            373450   8.0500   NaN        S  
	5      0            330877   8.4583   NaN        Q  
	6      0             17463  51.8625   E46        S  
	7      1            349909  21.0750   NaN        S  
	8      2            347742  11.1333   NaN        S  
	9      0            237736  30.0708   NaN        C  


## Descriptive statistics

### counting and summarizing a data series
counting the frequencies of each unique element in the data series
	
	In [31]: age_counts = train_age.value_counts()
look at the first 10 records of the count results
	
	In [32]: age_counts.head(10)
	Out[32]: 
	24    30
	22    27
	18    26
	28    25
	30    25
	19    25
	21    24
	25    23
	36    22
	29    20
	dtype: int64

The results are in descending order by default. In this special case, the age_counts data series has integer values. We can show all the records all the way to a specific value (say age 10 in this case) using

	In [33]: age_counts[:10]
	Out[33]: 
	24.0    30
	22.0    27
	18.0    26
	28.0    25
	30.0    25
	19.0    25
	...
	56.0     4
	8.0      4
	61.0     3
	6.0      3
	65.0     3
	7.0      3
	46.0     3
	32.5     2
	28.5     2
	40.5     2
	10.0     2
	dtype: int64

count the total and skip all the missing values:
	
	In [34]: train_age.sum(skipna = True)
	Out[34]: 21205.169999999998
wondering what happens if you don't skip missing values?
	
	In [35]: train_age.sum(skipna = False)
	Out[35]: nan

it won't calculate the sum because it can't do sum with unknown values! it's a safe way to take the sum and make sure that you are not missing anything.

we can also look at the oldest
	
	In [36]: train_age.max()
	Out[36]: 80.0

the youngest
	
	In [37]: train_age.min()
	Out[37]: 0.41999999999999998
	
the average
	
	In [38]: train_age.mean()
	Out[38]: 29.69911764705882

and the median
	
	In [39]: train_age.median()
	Out[39]: 28.0

Or we can use a default method to look at the summary statistics of age all together:
	
	In [40]: train_age.describe()
	Out[40]: 
	count    714.000000
	mean      29.699118
	std       14.526497
	min        0.420000
	25%       20.125000
	50%       28.000000
	75%       38.000000
	max       80.000000
	Name: Age, dtype: float64

describing the whole dataframe
	
	In [41]: train.describe()
	Out[41]: 
	       PassengerId    Survived      Pclass         Age       SibSp  \
	count   891.000000  891.000000  891.000000  714.000000  891.000000   
	mean    446.000000    0.383838    2.308642   29.699118    0.523008   
	std     257.353842    0.486592    0.836071   14.526497    1.102743   
	min       1.000000    0.000000    1.000000    0.420000    0.000000   
	25%     223.500000    0.000000    2.000000   20.125000    0.000000   
	50%     446.000000    0.000000    3.000000   28.000000    0.000000   
	75%     668.500000    1.000000    3.000000   38.000000    1.000000   
	max     891.000000    1.000000    3.000000   80.000000    8.000000   

	            Parch        Fare  
	count  891.000000  891.000000  
	mean     0.381594   32.204208  
	std      0.806057   49.693429  
	min      0.000000    0.000000  
	25%      0.000000    7.910400  
	50%      0.000000   14.454200  
	75%      0.000000   31.000000  
	max      6.000000  512.329200  


wait, where are the string variables? let's take a look at the ticket column:
	
	In [42]: train['Ticket'].describe()
	Out[42]: 
	count          891
	unique         681
	top       CA. 2343
	freq             7
	Name: Ticket, dtype: object

it turns out that '.describe' displays numeric variables if the dataframe has a mix of different types of varibles. To specify each type, we can use:
	
	In [43]: train.describe(include =['object'])
	Out[43]: 
	                             Name   Sex    Ticket        Cabin Embarked
	count                         891   891       891          204      889
	unique                        891     2       681          147        3
	top     Graham, Mr. George Edward  male  CA. 2343  C23 C25 C27        S
	freq                            1   577         7            4      644
	
	In [44]: train.describe(include =['number'])
	Out[44]: 
	       PassengerId    Survived      Pclass         Age       SibSp  \
	count   891.000000  891.000000  891.000000  714.000000  891.000000   
	mean    446.000000    0.383838    2.308642   29.699118    0.523008   
	std     257.353842    0.486592    0.836071   14.526497    1.102743   
	min       1.000000    0.000000    1.000000    0.420000    0.000000   
	25%     223.500000    0.000000    2.000000   20.125000    0.000000   
	50%     446.000000    0.000000    3.000000   28.000000    0.000000   
	75%     668.500000    1.000000    3.000000   38.000000    1.000000   
	max     891.000000    1.000000    3.000000   80.000000    8.000000   

	            Parch        Fare  
	count  891.000000  891.000000  
	mean     0.381594   32.204208  
	std      0.806057   49.693429  
	min      0.000000    0.000000  
	25%      0.000000    7.910400  
	50%      0.000000   14.454200  
	75%      0.000000   31.000000  
	max      6.000000  512.329200  

and to look at the whole thing (although some values won't show):
	
	In [45]: train.describe(include ='all')
	Out[45]: 
	        PassengerId    Survived      Pclass                       Name   Sex  \
	count    891.000000  891.000000  891.000000                        891   891   
	unique          NaN         NaN         NaN                        891     2   
	top             NaN         NaN         NaN  Graham, Mr. George Edward  male   
	freq            NaN         NaN         NaN                          1   577   
	mean     446.000000    0.383838    2.308642                        NaN   NaN   
	std      257.353842    0.486592    0.836071                        NaN   NaN   
	min        1.000000    0.000000    1.000000                        NaN   NaN   
	25%      223.500000    0.000000    2.000000                        NaN   NaN   
	50%      446.000000    0.000000    3.000000                        NaN   NaN   
	75%      668.500000    1.000000    3.000000                        NaN   NaN   
	max      891.000000    1.000000    3.000000                        NaN   NaN   

	               Age       SibSp       Parch    Ticket        Fare        Cabin  \
	count   714.000000  891.000000  891.000000       891  891.000000          204   
	unique         NaN         NaN         NaN       681         NaN          147   
	top            NaN         NaN         NaN  CA. 2343         NaN  C23 C25 C27   
	freq           NaN         NaN         NaN         7         NaN            4   
	mean     29.699118    0.523008    0.381594       NaN   32.204208          NaN   
	std      14.526497    1.102743    0.806057       NaN   49.693429          NaN   
	min       0.420000    0.000000    0.000000       NaN    0.000000          NaN   
	25%      20.125000    0.000000    0.000000       NaN    7.910400          NaN   
	50%      28.000000    0.000000    0.000000       NaN   14.454200          NaN   
	75%      38.000000    1.000000    0.000000       NaN   31.000000          NaN   
	max      80.000000    8.000000    6.000000       NaN  512.329200          NaN   

	       Embarked  
	count       889  
	unique        3  
	top           S  
	freq        644  
	mean        NaN  
	std         NaN  
	min         NaN  
	25%         NaN  
	50%         NaN  
	75%         NaN  
	max         NaN  


## Subsetting Data Using Criteria

### subsetting & filtering using Boolean values
pandas does vectorized calculation. This compares every value in train_age with 60, and generates a series of True or False
	
	In [46]: train_age > 60
	Out[46]: 
	0      False
	1      False
	2      False
	3      False
	4      False
	5      False
	       ...  
	881    False
	882    False
	883    False
	884    False
	885    False
	886    False
	887    False
	888    False
	889    False
	890    False
	Name: Age, dtype: bool


Applying the series the same way you select a column to select all the records in the age column at age above 60
	
	In [47]: train_age[train_age > 60].head()
	Out[47]: 
	33     66.0
	54     65.0
	96     71.0
	116    70.5
	170    61.0
	Name: Age, dtype: float64

we can also only select all the 60 yrs olds in the age column
	
	In [48]: train_age[train.Age == 60]
	Out[48]: 
	366    60
	587    60
	684    60
	694    60
	Name: Age, dtype: float64

it's boring to look at only ages, so let's select the whole dataframe using the same criteria, again the first 5 rows
	
	In [49]: train[train_age == 60].head()
	Out[49]: 
	     PassengerId  Survived  Pclass  \
	366          367         1       1   
	587          588         1       1   
	684          685         0       2   
	694          695         0       1   

	                                                 Name     Sex  Age  SibSp  \
	366  Warren, Mrs. Frank Manley (Anna Sophia Atkinson)  female   60      1   
	587                  Frolicher-Stehli, Mr. Maxmillian    male   60      1   
	684                 Brown, Mr. Thomas William Solomon    male   60      1   
	694                                   Weir, Col. John    male   60      0   

	     Parch  Ticket   Fare Cabin Embarked  
	366      0  110813  75.25   D37        C  
	587      1   13567  79.20   B41        C  
	684      1   29750  39.00   NaN        S  
	694      0  113800  26.55   NaN        S  


We can use greater than 70 as a filter on all columns
	
	In [50]: train[train > 70][:10]
	Out[50]: 
	   PassengerId  Survived  Pclass  \
	0          NaN       NaN     NaN   
	1          NaN       NaN     NaN   
	2          NaN       NaN     NaN   
	3          NaN       NaN     NaN   
	4          NaN       NaN     NaN   
	5          NaN       NaN     NaN   
	6          NaN       NaN     NaN   
	7          NaN       NaN     NaN   
	8          NaN       NaN     NaN   
	9          NaN       NaN     NaN   

	                                                Name     Sex  Age  SibSp  \
	0                            Braund, Mr. Owen Harris    male  NaN    NaN   
	1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  NaN    NaN   
	2                             Heikkinen, Miss. Laina  female  NaN    NaN   
	3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  NaN    NaN   
	4                           Allen, Mr. William Henry    male  NaN    NaN   
	5                                   Moran, Mr. James    male  NaN    NaN   
	6                            McCarthy, Mr. Timothy J    male  NaN    NaN   
	7                     Palsson, Master. Gosta Leonard    male  NaN    NaN   
	8  Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)  female  NaN    NaN   
	9                Nasser, Mrs. Nicholas (Adele Achem)  female  NaN    NaN   

	   Parch            Ticket     Fare Cabin Embarked  
	0    NaN         A/5 21171      NaN   NaN        S  
	1    NaN          PC 17599  71.2833   C85        C  
	2    NaN  STON/O2. 3101282      NaN   NaN        S  
	3    NaN            113803      NaN  C123        S  
	4    NaN            373450      NaN   NaN        S  
	5    NaN            330877      NaN   NaN        Q  
	6    NaN             17463      NaN   E46        S  
	7    NaN            349909      NaN   NaN        S  
	8    NaN            347742      NaN   NaN        S  
	9    NaN            237736      NaN   NaN        C  

or use the .where() function

	In [51]: train.where(train > 70)[:10]
	Out[51]: 
	   PassengerId  Survived  Pclass  \
	0          NaN       NaN     NaN   
	1          NaN       NaN     NaN   
	2          NaN       NaN     NaN   
	3          NaN       NaN     NaN   
	4          NaN       NaN     NaN   
	5          NaN       NaN     NaN   
	6          NaN       NaN     NaN   
	7          NaN       NaN     NaN   
	8          NaN       NaN     NaN   
	9          NaN       NaN     NaN   

	                                                Name     Sex  Age  SibSp  \
	0                            Braund, Mr. Owen Harris    male  NaN    NaN   
	1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  NaN    NaN   
	2                             Heikkinen, Miss. Laina  female  NaN    NaN   
	3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  NaN    NaN   
	4                           Allen, Mr. William Henry    male  NaN    NaN   
	5                                   Moran, Mr. James    male  NaN    NaN   
	6                            McCarthy, Mr. Timothy J    male  NaN    NaN   
	7                     Palsson, Master. Gosta Leonard    male  NaN    NaN   
	8  Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)  female  NaN    NaN   
	9                Nasser, Mrs. Nicholas (Adele Achem)  female  NaN    NaN   

	   Parch            Ticket     Fare Cabin Embarked  
	0    NaN         A/5 21171      NaN   NaN        S  
	1    NaN          PC 17599  71.2833   C85        C  
	2    NaN  STON/O2. 3101282      NaN   NaN        S  
	3    NaN            113803      NaN  C123        S  
	4    NaN            373450      NaN   NaN        S  
	5    NaN            330877      NaN   NaN        Q  
	6    NaN             17463      NaN   E46        S  
	7    NaN            349909      NaN   NaN        S  
	8    NaN            347742      NaN   NaN        S  
	9    NaN            237736      NaN   NaN        C  


select all the people at age not equal to 60
	
	In [52]: train[train_age != 60]
	Out[52]: 
	     PassengerId  Survived  Pclass  \
	0              1         0       3   
	1              2         1       1   
	2              3         1       3   
	3              4         1       1   
	..           ...       ...     ...   
	886          887         0       2   
	887          888         1       1   
	888          889         0       3   
	889          890         1       1   
	890          891         0       3 
		 Name     Sex  Age  SibSp  \
	0                              Braund, Mr. Owen Harris    male   22      1   
	1    Cumings, Mrs. John Bradley (Florence Briggs Th...  female   38      1   
	2                               Heikkinen, Miss. Laina  female   26      0   
	3         Futrelle, Mrs. Jacques Heath (Lily May Peel)  female   35      1   
	4                             Allen, Mr. William Henry    male   35      0   
	..                                                 ...     ...  ...    ...   
	886                              Montvila, Rev. Juozas    male   27      0   
	887                       Graham, Miss. Margaret Edith  female   19      0   
	888           Johnston, Miss. Catherine Helen "Carrie"  female  NaN      1   
	889                              Behr, Mr. Karl Howell    male   26      0   
	890                                Dooley, Mr. Patrick    male   32      0   
	   Parch            Ticket      Fare        Cabin Embarked  
	0        0         A/5 21171    7.2500          NaN        S  
	1        0          PC 17599   71.2833          C85        C  
	2        0  STON/O2. 3101282    7.9250          NaN        S  
	3        0            113803   53.1000         C123        S  
	4        0            373450    8.0500          NaN        S  
	..     ...               ...       ...          ...      ... 
	887      0            112053   30.0000          B42        S  
	888      2        W./C. 6607   23.4500          NaN        S  
	889      0            111369   30.0000         C148        C  
	890      0            370376    7.7500          NaN        Q  

	[887 rows x 12 columns]
 
or reverse that (equivalent to train[train_age == 60])
	
	In [53]: train[~(train_age != 60)]
	Out[53]: 
	     PassengerId  Survived  Pclass  \
	366          367         1       1   
	587          588         1       1   
	684          685         0       2   
	694          695         0       1   

	                                                 Name     Sex  Age  SibSp  \
	366  Warren, Mrs. Frank Manley (Anna Sophia Atkinson)  female   60      1   
	587                  Frolicher-Stehli, Mr. Maxmillian    male   60      1   
	684                 Brown, Mr. Thomas William Solomon    male   60      1   
	694                                   Weir, Col. John    male   60      0   

	     Parch  Ticket   Fare Cabin Embarked  
	366      0  110813  75.25   D37        C  
	587      1   13567  79.20   B41        C  
	684      1   29750  39.00   NaN        S  
	694      0  113800  26.55   NaN        S  


### using multiple criteria
age between 60 and 75
	
	In [54]: train[(train_age > 60) & (train_age < 75)]
	Out[54]: 
	     PassengerId  Survived  Pclass                                       Name  \
	33            34         0       2                      Wheadon, Mr. Edward H   
	54            55         0       1             Ostby, Mr. Engelhart Cornelius   
	96            97         0       1                  Goldschmidt, Mr. George B   
	116          117         0       3                       Connors, Mr. Patrick   
	170          171         0       1                  Van der hoef, Mr. Wyckoff   
	252          253         0       1                  Stead, Mr. William Thomas   
	275          276         1       1          Andrews, Miss. Kornelia Theodosia   
	280          281         0       3                           Duane, Mr. Frank   
	326          327         0       3                  Nysveen, Mr. Johan Hansen   
	438          439         0       1                          Fortune, Mr. Mark   
	456          457         0       1                  Millet, Mr. Francis Davis   
	483          484         1       3                     Turkula, Mrs. (Hedwig)   
	493          494         0       1                    Artagaveytia, Mr. Ramon   
	545          546         0       1               Nicholson, Mr. Arthur Ernest   
	555          556         0       1                         Wright, Mr. George   
	570          571         1       2                         Harris, Mr. George   
	625          626         0       1                      Sutton, Mr. Frederick   
	672          673         0       2                Mitchell, Mr. Henry Michael   
	745          746         0       1               Crosby, Capt. Edward Gifford   
	829          830         1       1  Stone, Mrs. George Nelson (Martha Evelyn)   
	851          852         0       3                        Svensson, Mr. Johan   

	        Sex   Age  SibSp  Parch       Ticket      Fare        Cabin Embarked  
	33     male  66.0      0      0   C.A. 24579   10.5000          NaN        S  
	54     male  65.0      0      1       113509   61.9792          B30        C  
	96     male  71.0      0      0     PC 17754   34.6542           A5        C  
	116    male  70.5      0      0       370369    7.7500          NaN        Q  
	170    male  61.0      0      0       111240   33.5000          B19        S  
	252    male  62.0      0      0       113514   26.5500          C87        S  
	275  female  63.0      1      0        13502   77.9583           D7        S  
	280    male  65.0      0      0       336439    7.7500          NaN        Q  
	326    male  61.0      0      0       345364    6.2375          NaN        S  
	438    male  64.0      1      4        19950  263.0000  C23 C25 C27        S  
	456    male  65.0      0      0        13509   26.5500          E38        S  
	483  female  63.0      0      0         4134    9.5875          NaN        S  
	493    male  71.0      0      0     PC 17609   49.5042          NaN        C  
	545    male  64.0      0      0          693   26.0000          NaN        S  
	555    male  62.0      0      0       113807   26.5500          NaN        S  
	570    male  62.0      0      0  S.W./PP 752   10.5000          NaN        S  
	625    male  61.0      0      0        36963   32.3208          D50        S  
	672    male  70.0      0      0   C.A. 24580   10.5000          NaN        S  
	745    male  70.0      1      1    WE/P 5735   71.0000          B22        S  
	829  female  62.0      0      0       113572   80.0000          B28      NaN  
	851    male  74.0      0      0       347060    7.7750          NaN        S 

age above 75 or below 2
	
	In [55]: train[(train_age > 75) | (train_age < 2)]
	Out[55]: 
	     PassengerId  Survived  Pclass                                  Name  \
	78            79         1       2         Caldwell, Master. Alden Gates   
	164          165         0       3          Panula, Master. Eino Viljami   
	172          173         1       3          Johnson, Miss. Eleanor Ileen   
	183          184         1       2             Becker, Master. Richard F   
	305          306         1       1        Allison, Master. Hudson Trevor   
	381          382         1       3           Nakid, Miss. Maria ("Mary")   
	386          387         0       3       Goodwin, Master. Sidney Leonard   
	469          470         1       3         Baclini, Miss. Helene Barbara   
	630          631         1       1  Barkworth, Mr. Algernon Henry Wilson   
	644          645         1       3                Baclini, Miss. Eugenie   
	755          756         1       2             Hamalainen, Master. Viljo   
	788          789         1       3            Dean, Master. Bertram Vere   
	803          804         1       3       Thomas, Master. Assad Alexander   
	827          828         1       2                 Mallet, Master. Andre   
	831          832         1       2       Richards, Master. George Sibley   

	        Sex    Age  SibSp  Parch           Ticket      Fare    Cabin Embarked  
	78     male   0.83      0      2           248738   29.0000      NaN        S  
	164    male   1.00      4      1          3101295   39.6875      NaN        S  
	172  female   1.00      1      1           347742   11.1333      NaN        S  
	183    male   1.00      2      1           230136   39.0000       F4        S  
	305    male   0.92      1      2           113781  151.5500  C22 C26        S  
	381  female   1.00      0      2             2653   15.7417      NaN        C  
	386    male   1.00      5      2          CA 2144   46.9000      NaN        S  
	469  female   0.75      2      1             2666   19.2583      NaN        C  
	630    male  80.00      0      0            27042   30.0000      A23        S  
	644  female   0.75      2      1             2666   19.2583      NaN        C  
	755    male   0.67      1      1           250649   14.5000      NaN        S  
	788    male   1.00      1      2        C.A. 2315   20.5750      NaN        S  
	803    male   0.42      0      1             2625    8.5167      NaN        C  
	827    male   1.00      0      2  S.C./PARIS 2079   37.0042      NaN        C  
	831    male   0.83      1      1            29106   18.7500      NaN        S 

### Selecting Challenge: 

Select all the people that are:

1. in the 3rd class
2. under age 10
3. female 

Anwser: 

	train[(train['Age'] < 10) & (train['Pclass'] == 3) & (train['Sex'] == "female")]

A different way to match multiple criteria using 'in'
	
	In [57]: train[train_age.isin([20,30])]
	Out[57]: 
	     PassengerId  Survived  Pclass  \
	12            13         0       3   
	79            80         1       3   
	91            92         0       3   
	113          114         0       3   
	131          132         0       3   
	157          158         0       3   
	178          179         0       2   
	213          214         0       2   
	219          220         0       2   
	244          245         0       3   
	253          254         0       3   
	257          258         1       1   
	286          287         1       3   
	308          309         0       2   
	309          310         1       1   
	322          323         1       2   
	365          366         0       3   
	378          379         0       3   
	404          405         0       3   
	418          419         0       2   
	441          442         0       3   
	452          453         0       1   
	488          489         0       3   
	520          521         1       1   
	534          535         0       3   
	537          538         1       1   
	606          607         0       3   
	622          623         1       3   
	640          641         0       3   
	664          665         1       3   
	682          683         0       3   
	725          726         0       3   
	726          727         1       2   
	747          748         1       2   
	762          763         1       3   
	798          799         0       3   
	799          800         0       3   
	840          841         0       3   
	842          843         1       1   
	876          877         0       3   

	                                                  Name     Sex  Age  SibSp  \
	12                      Saundercock, Mr. William Henry    male   20      0   
	79                            Dowdell, Miss. Elizabeth  female   30      0   
	91                          Andreasson, Mr. Paul Edvin    male   20      0   
	113                            Jussila, Miss. Katriina  female   20      1   
	131                     Coelho, Mr. Domingos Fernandeo    male   20      0   
	157                                    Corn, Mr. Harry    male   30      0   
	178                                 Hale, Mr. Reginald    male   30      0   
	213                        Givard, Mr. Hans Kristensen    male   30      0   
	219                                 Harris, Mr. Walter    male   30      0   
	244                               Attalah, Mr. Sleiman    male   30      0   
	253                           Lobb, Mr. William Arthur    male   30      1   
	257                               Cherry, Miss. Gladys  female   30      0   
	286                            de Mulder, Mr. Theodore    male   30      0   
	308                                Abelson, Mr. Samuel    male   30      1   
	309                     Francatelli, Miss. Laura Mabel  female   30      0   
	322                          Slayter, Miss. Hilda Mary  female   30      0   
	365                     Adahl, Mr. Mauritz Nils Martin    male   30      0   
	378                                Betros, Mr. Tannous    male   20      0   
	404                            Oreskovic, Miss. Marija  female   20      0   
	418                         Matthews, Mr. William John    male   30      0   
	441                                    Hampe, Mr. Leon    male   20      0   
	452                    Foreman, Mr. Benjamin Laventall    male   30      0   
	488                      Somerton, Mr. Francis William    male   30      0   
	520                              Perreault, Miss. Anne  female   30      0   
	534                                Cacic, Miss. Marija  female   30      0   
	537                                LeRoy, Miss. Bertha  female   30      0   
	606                                  Karaic, Mr. Milan    male   30      0   
	622                                   Nakid, Mr. Sahid    male   20      1   
	640                             Jensen, Mr. Hans Peder    male   20      0   
	664                        Lindqvist, Mr. Eino William    male   20      1   
	682                        Olsvigen, Mr. Thor Anderson    male   20      0   
	725                                Oreskovic, Mr. Luka    male   20      0   
	726        Renouf, Mrs. Peter Henry (Lillian Jefferys)  female   30      3   
	747                              Sinkkonen, Miss. Anna  female   30      0   
	762                              Barah, Mr. Hanna Assi    male   20      0   
	798                       Ibrahim Shawah, Mr. Yousseff    male   30      0   
	799  Van Impe, Mrs. Jean Baptiste (Rosalie Paula Go...  female   30      1   
	840                        Alhomaki, Mr. Ilmari Rudolf    male   20      0   
	842                            Serepeca, Miss. Augusta  female   30      0   
	876                      Gustafsson, Mr. Alfred Ossian    male   20      0   

	     Parch              Ticket      Fare Cabin Embarked  
	12       0           A/5. 2151    8.0500   NaN        S  
	79       0              364516   12.4750   NaN        S  
	91       0              347466    7.8542   NaN        S  
	113      0                4136    9.8250   NaN        S  
	131      0  SOTON/O.Q. 3101307    7.0500   NaN        S  
	157      0     SOTON/OQ 392090    8.0500   NaN        S  
	178      0              250653   13.0000   NaN        S  
	213      0              250646   13.0000   NaN        S  
	219      0           W/C 14208   10.5000   NaN        S  
	244      0                2694    7.2250   NaN        C  
	253      0           A/5. 3336   16.1000   NaN        S  
	257      0              110152   86.5000   B77        S  
	286      0              345774    9.5000   NaN        S  
	308      0           P/PP 3381   24.0000   NaN        C  
	309      0            PC 17485   56.9292   E36        C  
	322      0              234818   12.3500   NaN        Q  
	365      0              C 7076    7.2500   NaN        S  
	378      0                2648    4.0125   NaN        C  
	404      0              315096    8.6625   NaN        S  
	418      0               28228   13.0000   NaN        S  
	441      0              345769    9.5000   NaN        S  
	452      0              113051   27.7500  C111        C  
	488      0          A.5. 18509    8.0500   NaN        S  
	520      0               12749   93.5000   B73        S  
	534      0              315084    8.6625   NaN        S  
	537      0            PC 17761  106.4250   NaN        C  
	606      0              349246    7.8958   NaN        S  
	622      1                2653   15.7417   NaN        C  
	640      0              350050    7.8542   NaN        S  
	664      0   STON/O 2. 3101285    7.9250   NaN        S  
	682      0                6563    9.2250   NaN        S  
	725      0              315094    8.6625   NaN        S  
	726      0               31027   21.0000   NaN        S  
	747      0              250648   13.0000   NaN        S  
	762      0                2663    7.2292   NaN        C  
	798      0                2685    7.2292   NaN        C  
	799      1              345773   24.1500   NaN        S  
	840      0    SOTON/O2 3101287    7.9250   NaN        S  
	842      0              113798   31.0000   NaN        C  
	876      0                7534    9.8458   NaN        S  


	train[(train['Age'] < 10) & (train['pclass'] == 3) & (train['sex'] == "female")]

A different way to match multiple criteria using 'in'
	
	train[train_age.isin([20,30])]

and 'not in' using the '~' in from of the criteria
	
	train[~train_age.isin([20,30])]

	[851 rows x 12 columns]

It works on strings too:
	
	In [59]: train[train['Ticket'].isin(['113803'])]
	Out[59]: 
	     PassengerId  Survived  Pclass  \
	3              4         1       1   
	137          138         0       1   

	                                             Name     Sex  Age  SibSp  Parch  \
	3    Futrelle, Mrs. Jacques Heath (Lily May Peel)  female   35      1      0   
	137                   Futrelle, Mr. Jacques Heath    male   37      1      0   

	     Ticket  Fare Cabin Embarked  
	3    113803  53.1  C123        S  
	137  113803  53.1  C123        S  



To select all tickets that match 'PC 17599':
	
	In [60]: train['Ticket'][train['Ticket'].str.match('PC 17599', as_indexer=True)]Out[60]: 
	1    PC 17599
	Name: Ticket, dtype: object
or to get all the records with the same criteria:
	
	In [61]: train[train['Ticket'].str.match('PC 17599', as_indexer=True)]
	Out[61]: 
	   PassengerId  Survived  Pclass  \
	1            2         1       1   

	                                                Name     Sex  Age  SibSp  \
	1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female   38      1   

	   Parch    Ticket     Fare Cabin Embarked  
	1      0  PC 17599  71.2833   C85        C  

We can use 'contains' to find names that has 'Lily' in them
	
	In [62]: train['Name'][train['Name'].str.contains('Lily')]
	Out[62]: 
	3       Futrelle, Mrs. Jacques Heath (Lily May Peel)
	879    Potter, Mrs. Thomas Jr (Lily Alexenia Wilson)
	Name: Name, dtype: object


	
'startswith' and 'endswith'
	
	In [63]: train['Name'][train['Name'].str.endswith('Peel)')]
	Out[63]: 
	3    Futrelle, Mrs. Jacques Heath (Lily May Peel)
	Name: Name, dtype: object
	
	In [64]: train['Name'][train['Name'].str.startswith('Futrelle')]
	Out[64]: 
	3      Futrelle, Mrs. Jacques Heath (Lily May Peel)
	137                     Futrelle, Mr. Jacques Heath
	Name: Name, dtype: object

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
	

	In [65]: train_age.order()
	Out[65]: 
	803    0.42
	755    0.67
	644    0.75
	469    0.75
	78     0.83
	831    0.83
	305    0.92
	       ... 
	859     NaN
	863     NaN
	868     NaN
	878     NaN
	888     NaN
	Name: Age, dtype: float64
reverse the order
	
	In [66]: train_age.order(ascending = False)
	Out[66]: 
	630    80.0
	851    74.0
	96     71.0
	493    71.0
	116    70.5
	745    70.0
	       ... 
	859     NaN
	863     NaN
	868     NaN
	878     NaN
	888     NaN
	Name: Age, dtype: float64
the first 3 smallest numbers in the age data series
	
	In [67]: train_age.nsmallest(3)
	Out[67]: 
	803    0.42
	755    0.67
	469    0.75
	Name: Age, dtype: float64

the first 3 largest numbers in the age data series
	
	In [68]: train_age.nlargest(3)
	Out[68]: 
	630    80
	851    74
	96     71
	Name: Age, dtype: float64

### sort by multi-indexing
sort the dataframe by age and show the top 5 results
	
	In [70]: train.sort_index(by = 'Age').head()
	Out[70]: 
	     PassengerId  Survived  Pclass                             Name     Sex  \
	803          804         1       3  Thomas, Master. Assad Alexander    male   
	755          756         1       2        Hamalainen, Master. Viljo    male   
	644          645         1       3           Baclini, Miss. Eugenie  female   
	469          470         1       3    Baclini, Miss. Helene Barbara  female   
	78            79         1       2    Caldwell, Master. Alden Gates    male   

	      Age  SibSp  Parch  Ticket     Fare Cabin Embarked  
	803  0.42      0      1    2625   8.5167   NaN        C  
	755  0.67      1      1  250649  14.5000   NaN        S  
	644  0.75      2      1    2666  19.2583   NaN        C  
	469  0.75      2      1    2666  19.2583   NaN        C  
	78   0.83      0      2  248738  29.0000   NaN        S  



sort the dataframe first by age, then pclass
	
	In [71]: train.sort_index(by = ['Age','Pclass'])[:10]
	Out[71]: 
	     PassengerId  Survived  Pclass                             Name     Sex  \
	803          804         1       3  Thomas, Master. Assad Alexander    male   
	755          756         1       2        Hamalainen, Master. Viljo    male   
	469          470         1       3    Baclini, Miss. Helene Barbara  female   
	644          645         1       3           Baclini, Miss. Eugenie  female   
	78            79         1       2    Caldwell, Master. Alden Gates    male   
	831          832         1       2  Richards, Master. George Sibley    male   
	305          306         1       1   Allison, Master. Hudson Trevor    male   
	183          184         1       2        Becker, Master. Richard F    male   
	827          828         1       2            Mallet, Master. Andre    male   
	164          165         0       3     Panula, Master. Eino Viljami    male   

	      Age  SibSp  Parch           Ticket      Fare    Cabin Embarked  
	803  0.42      0      1             2625    8.5167      NaN        C  
	755  0.67      1      1           250649   14.5000      NaN        S  
	469  0.75      2      1             2666   19.2583      NaN        C  
	644  0.75      2      1             2666   19.2583      NaN        C  
	78   0.83      0      2           248738   29.0000      NaN        S  
	831  0.83      1      1            29106   18.7500      NaN        S  
	305  0.92      1      2           113781  151.5500  C22 C26        S  
	183  1.00      2      1           230136   39.0000       F4        S  
	827  1.00      0      2  S.C./PARIS 2079   37.0042      NaN        C  
	164  1.00      4      1          3101295   39.6875      NaN        S  

sort the dataframe first by class, then age (the order of column names determines which one gets sorted first)
	
	In [72]: train.sort_index(by = ['Pclass','Age'])[:10]
	Out[72]: 
	     PassengerId  Survived  Pclass  \
	305          306         1       1   
	297          298         0       1   
	445          446         1       1   
	802          803         1       1   
	435          436         1       1   
	689          690         1       1   
	329          330         1       1   
	504          505         1       1   
	853          854         1       1   
	307          308         1       1   

	                                                  Name     Sex    Age  SibSp  \
	305                     Allison, Master. Hudson Trevor    male   0.92      1   
	297                       Allison, Miss. Helen Loraine  female   2.00      1   
	445                          Dodge, Master. Washington    male   4.00      0   
	802                Carter, Master. William Thornton II    male  11.00      1   
	435                          Carter, Miss. Lucile Polk  female  14.00      1   
	689                  Madill, Miss. Georgette Alexandra  female  15.00      0   
	329                       Hippach, Miss. Jean Gertrude  female  16.00      0   
	504                              Maioni, Miss. Roberta  female  16.00      0   
	853                          Lines, Miss. Mary Conover  female  16.00      0   
	307  Penasco y Castellana, Mrs. Victor de Satode (M...  female  17.00      1   

	     Parch    Ticket      Fare    Cabin Embarked  
	305      2    113781  151.5500  C22 C26        S  
	297      2    113781  151.5500  C22 C26        S  
	445      2     33638   81.8583      A34        S  
	802      2    113760  120.0000  B96 B98        S  
	435      2    113760  120.0000  B96 B98        S  
	689      1     24160  211.3375       B5        S  
	329      1    111361   57.9792      B18        C  
	504      0    110152   86.5000      B79        S  
	853      1  PC 17592   39.4000      D28        S  
	307      0  PC 17758  108.9000      C65        C  

sort the dataframe first by class in descending order, then age in ascending order

	In [73]: train.sort_index(by = ['Pclass','Age'], ascending = [False,True])[:10]
	Out[73]: 
	     PassengerId  Survived  Pclass                             Name     Sex  \
	803          804         1       3  Thomas, Master. Assad Alexander    male   
	469          470         1       3    Baclini, Miss. Helene Barbara  female   
	644          645         1       3           Baclini, Miss. Eugenie  female   
	164          165         0       3     Panula, Master. Eino Viljami    male   
	172          173         1       3     Johnson, Miss. Eleanor Ileen  female   
	381          382         1       3      Nakid, Miss. Maria ("Mary")  female   
	386          387         0       3  Goodwin, Master. Sidney Leonard    male   
	788          789         1       3       Dean, Master. Bertram Vere    male   
	7              8         0       3   Palsson, Master. Gosta Leonard    male   
	16            17         0       3             Rice, Master. Eugene    male   

	      Age  SibSp  Parch     Ticket     Fare Cabin Embarked  
	803  0.42      0      1       2625   8.5167   NaN        C  
	469  0.75      2      1       2666  19.2583   NaN        C  
	644  0.75      2      1       2666  19.2583   NaN        C  
	164  1.00      4      1    3101295  39.6875   NaN        S  
	172  1.00      1      1     347742  11.1333   NaN        S  
	381  1.00      0      2       2653  15.7417   NaN        C  
	386  1.00      5      2    CA 2144  46.9000   NaN        S  
	788  1.00      1      2  C.A. 2315  20.5750   NaN        S  
	7    2.00      3      1     349909  21.0750   NaN        S  
	16   2.00      4      1     382652  29.1250   NaN        Q  


## missing values
we can check missing values using the following methods
	
	isnull()
	notnull()

The first 5 missing values in the age dataseries

	In [75]: train_age[train_age.isnull()].head()
	Out[75]: 
	5    NaN
	17   NaN
	19   NaN
	26   NaN
	28   NaN
	Name: Age, dtype: float64

The first 5 records that have missing age values
	
	In [76]: train[train_age.isnull()].head()
	Out[76]: 
	    PassengerId  Survived  Pclass                           Name     Sex  Age  \
	5             6         0       3               Moran, Mr. James    male  NaN   
	17           18         1       2   Williams, Mr. Charles Eugene    male  NaN   
	19           20         1       3        Masselmani, Mrs. Fatima  female  NaN   
	26           27         0       3        Emir, Mr. Farred Chehab    male  NaN   
	28           29         1       3  O'Dwyer, Miss. Ellen "Nellie"  female  NaN   

	    SibSp  Parch  Ticket     Fare Cabin Embarked  
	5       0      0  330877   8.4583   NaN        Q  
	17      0      0  244373  13.0000   NaN        S  
	19      0      0    2649   7.2250   NaN        C  
	26      0      0    2631   7.2250   NaN        C  
	28      0      0  330959   7.8792   NaN        Q  


You can replace missing values with anything using '.fillna()', but we don't recommend it! Missing values are better left as is, because programming languages usually treat them differently from everything else. If you replace them with any value, they may affect your calculations later on.
If you really have to do it, at least make a copy of the dataframe first:
	

	In [77]: train2 = train.copy()
then replace all missing age values with 0
	
	In [78]: train2['Age'] = train2['Age'].fillna(0)

to contrast with results from Out[75] above:
	In [81]: train2['Age'][[5,17,19,26,28]]
	Out[81]: 
	5     0
	17    0
	19    0
	26    0
	28    0
	Name: Age, dtype: object


We can also replace missing age values with strings
	
	In [82]: train3 = train.copy()
	In [83]: train3['Age'] = train3['Age'].fillna('wow')
	In [84]: train3['Age'][[5,17,19,26,28]]
	Out[84]: 
	5     wow
	17    wow
	19    wow
	26    wow
	28    wow
	Name: Age, dtype: object

Sometimes we may want to remove records that have missing values.We can use '.dropna()' to do that. 
Caution! Removing records with missing values will have serious consequences, so think about it twice before doing it!
	

	In [85]: train4 = train.copy()

	In [86]: train4 = train4.dropna()


## adding columns
We can add new columns to the dataframe by directly assigning values to the new column:
	

	In [87]: train['Gender'] = 4
	In [88]: train.dtypes
	Out[88]: 
	PassengerId      int64
	Survived         int64
	Pclass           int64
	Name            object
	Sex             object
	Age            float64
	SibSp            int64
	Parch            int64
	Ticket          object
	Fare           float64
	Cabin           object
	Embarked        object
	Gender           int64
	dtype: object


We can also add new columns using calculations:
	
	In [89]: train['Fare_to_age'] = train['Fare'] / train['Age']
	In [90]: train['Fare_to_age'].head()
	Out[90]: 
	0    0.329545
	1    1.875876
	2    0.304808
	3    1.517143
	4    0.230000
	Name: Fare_to_age, dtype: float64

We can also create columns with missing values:
	
	In [91]: import numpy as np
	In [92]: train['Missing'] = np.nan

replace 0 with missing values back too:
	
	In [93]: train2['Age'] = train2['Age'].replace(0, np.nan)
	In [94]: train2.Age[train2['Age'].isnull()].head()
	Out[94]: 
	5    NaN
	17   NaN
	19   NaN
	26   NaN
	28   NaN
	Name: Age, dtype: float64


## Applying elementwise Python functions in a data series
Adding a column called 'Gender' that is filled by the first letter from the values in 'Sex' column
	

	In [95]: train['Gender'] = train['Sex'].map(lambda x: x[0].upper())

	In [96]: train['Gender'].head()
	Out[96]: 
	0    M
	1    F
	2    F
	3    F
	4    M
	Name: Gender, dtype: object

If we want to apply a function in a dataframe, we use 

	.applymap
This function turns all the values into a string first, then replace its value with its string length
	
	In [97]: f = lambda x: len(str(x))

Then we apply the function to every value in the dataframe!
	
	In [98]: train.applymap(f)
	Out[98]: 
	     PassengerId  Survived  Pclass  Name  Sex  Age  SibSp  Parch  Ticket  \
	0              1         1       1    23    4    4      1      1       9   
	1              1         1       1    51    6    4      1      1       8   
	2              1         1       1    22    6    4      1      1      16   
	3              1         1       1    44    6    4      1      1       6   
	4              1         1       1    24    4    4      1      1       6   
	5              1         1       1    16    4    3      1      1       6 
		Fare  Cabin  Embarked  Gender  Fare_to_age  Missing  
	0       4      3         1       1           14        3  
	1       7      3         1       1           13        3  
	2       5      3         1       1           14        3  
	3       4      4         1       1           13        3  
	4       4      3         1       1            4        3  
	5       6      3         1       1            3        3  
	[891 rows x 15 columns]
If we don't assign the new values to the dataframe, it won't change the original:
	
	In [99]: train.head()
	Out[99]: 
	   PassengerId  Survived  Pclass  \
	0            1         0       3   
	1            2         1       1   
	2            3         1       3   
	3            4         1       1   
	4            5         0       3   

	                                                Name     Sex  Age  SibSp  \
	0                            Braund, Mr. Owen Harris    male   22      1   
	1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female   38      1   
	2                             Heikkinen, Miss. Laina  female   26      0   
	3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female   35      1   
	4                           Allen, Mr. William Henry    male   35      0   

	   Parch            Ticket     Fare Cabin Embarked Gender  Fare_to_age  \
	0      0         A/5 21171   7.2500   NaN        S      M     0.329545   
	1      0          PC 17599  71.2833   C85        C      F     1.875876   
	2      0  STON/O2. 3101282   7.9250   NaN        S      F     0.304808   
	3      0            113803  53.1000  C123        S      F     1.517143   
	4      0            373450   8.0500   NaN        S      M     0.230000   

	   Missing  
	0      NaN  
	1      NaN  
	2      NaN  
	3      NaN  
	4      NaN  


But if we do this, we replace the original dataframe with new values, and there is no way back unless we re-read the data file again:
	
	In [100]: train = train.head()

We can write the new dataframe out to a csv file
	
	In [101]: train.to_csv('./data/train_crazy.csv')

	In [102]: ls data/
	train.csv        train_crazy.csv

