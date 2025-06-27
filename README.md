# numpy-and-pandas
all practices  question of numpy and pandas  wiith detail 






# ============================  numpy array   are created===========================
#
# import numpy as np
# a=np.array([1,2,3,4,5,6,7])
# print(type(a))
#


# ===================  2D array  =======================================

# import numpy as np
# a=np.array([[1,2,3,4,5],[6,7,8,9,10]])
# print(a)


# =======================   3D   array  ==========================


# import numpy as np
# c=([[[1,2],[3,4]],[[5,6],[7,8]]])
# print(c)



# ===================   how to change the datatype of the  array  ====================

#
# import numpy as np
# d=np.array([1,2,3,4,5],dtype=bool)
# print(d)


# ========================  arange array  =============================================
#
# import numpy as np
# a=np.arange(1,11)
# print(a)



# ================================  reshape ==================
#
# import numpy as np
# a=np.arange(1,13).reshape(2,5)
# print(a)



# ========================  ones  ==============================


# import numpy as np
# a=np.ones((3,4))
# print(a)










# ==============================  zero  =========================

#
# import numpy as np
# a=np.zeros((3,4))
# print(a)



# =========================  random    =================================

#
# import numpy as np
# a=np.random.random((3,4))
# print(a)
#



# ==========================   linspace  =================================
#
# import numpy as np
# b=np.linspace(-10,10,7)
# print(b)



# =========================    identity  =================================
#
# import numpy as np
# a=np.identity(5)
# print(a)


# ==================   no of dimension   ===============================
#
# import  numpy as np
# a1=np.arange(10)
# a2=np.arange(12).reshape(4,3)
# a3=np.arange(16).reshape(2,2,2,2)
# print(a3)
#




# ==================================  shape  =================================

#
# import numpy as np
# a=np.arange(64,dtype=float).reshape(4,4,4)
# print(a)
# print("show item",a.shape)
# print("size of the array=",a.size)
# print("size of the array  in memory=  ",a.itemsize)
# print(a.dtype)
#
# print(a.astype(np.int32))

# ===========================   array operation ====================


# =====================    array function  ============================

# import numpy as np
# a=np.random.random((3,3))
# a=np.around(a*100)
# print(sum(a))
# print(np.sum(a, axis=1))



# ===================     statusial   function  ========================

#
# import numpy as np
# a=np.random.random((3,3))
# a=np.around(a*100)
# print(np.mean(a))


# =============================  dot product  ===========================
#
# import numpy as np
# a=np.arange(12).reshape(3,4)
# b=np.arange(12,24).reshape(4,3)
# print("a=",a)
# print("b=",b)
# print("dot product  =",np.dot(a, b))



# ========================   exponenet function ===================

#
#
# import numpy as np
# a=np.random.random((3,2))*100
# print(a)
# print(np.ceil(a))



#  =============================   indexng   in numpy  ====================
#
#
# import numpy as np
# a=np.arange(12).reshape(3,4)
# print(a)
# print(a[0,:])
# print(a[1:,1:3])
#
# print(a[::2,::3])
# print(a[1:3,1::2])
#
# print(a[::2,1::2])



# ============================    slices at the 3D array  ============================


# import numpy as np
# a3=np.arange(27).reshape(3,3,3)
# print(a3[::2])
# print(a3[0,1,:])
#
# print(a3[2,1:,1:])
#
# print(a3[::2,0,::2])




# =========================  iterating at the array  ===========================
#
# import numpy as np
# a=np.arange(27).reshape(3,3,3)
# print(a)
# for i in a:
#     print(i)

# =====================transpose   ====================================================
#
# import numpy as np
# a=np.arange(12).reshape(3,4)
# print(a)
# print(np.transpose(a))
# print(a.T)
#
# print("ravel array",a.ravel())


# =====================  strack  ====================================

#
# import numpy as np
# a1=np.arange(12).reshape(3,4)
# a2=np.arange(12,24).reshape(3,4)
# print(a1)
# print(a2)
#
# print(np.vsplit(a2,3))



# =========================== how to numpy is beter then the python  ===========================
#     ======================   in python =============================
# import sys
# a=[i for i in range(10000000)]
# print(getsizeof(a))



# =========================  in numpy  ================================
#
# import  numpy as np
# a=np.arange(10000000,dtype=np.int8)
# print(getsizeof(a))



# ===============  fency  indexing  =============================================
#
# import numpy as np
# a=np.arange(16).reshape(4,4)
# print(a[[0,2,3]])


# =====================  fency indexing in colum===============================
#
# import numpy as np
# a=np.arange(24).reshape(6,4)
# print(a)
# print(a[:,[0,2,3]])



# =======================  boolean indexing ===================================
# import numpy as np
# a=np.random.randint(1,100,24).reshape(6,4)
# print(a)
# print( a[(a > 50) & (a % 2 == 0)])



# ========================    broadcasting  =======================================

#
# from pandas.conftest import axis_1

# a=np.arange(12).reshape(4,3)
# b=np.arange(4)
# print(a)
# print(b)
# print(a+b)

#
# a=np.arange(1)
# b=np.arange(4).reshape(2,2)
# print(a+b)

#
# a=np.arange(10)
#
#
# def sigmoid(array):
#     return 1/(1+np.exp(array))
# print(sigmoid(a))
#


# ========================   mean squre error ===========================
#
# actual=np.random.randint(1,50,25)
# priductive=np.random.randint(1,50,25)
#
# def mse(actual,priductive):
#     return np.mean((actual-priductive)**2)
#
# print("actual data",actual)
# print("priductive data",priductive)
#
#
# print("find the mean square error =",mse(actual,priductive))


# ===================  missing value  =========================
#
# a=np.array([1,2,3,4,5,np.nan,7,8,9])
# print(a)
# print(a[~np.isnan(a)])


#
# ===================  ploting grape  ==========================
# import  matplotlib.pyplot as plt
# x=np.linspace(10,-10,100)
#
# y=np.sin(x)
#
# print(plt.plot(x,y))
# plt.show()


# ======================   sibmoid function =========================
#
# import matplotlib.pyplot as plt
#
# x=np.linspace(10,-10,100)
# y=1/(1+np.exp(-x))
# print(plt.plot(y))
# plt.show()



# # ======================   sorted array in numpy ===========================
#
# a=np.random.randint(1,100,15)
# print(a)
# print("sorted array",np.sort(a)[::-1])


#
# a=np.random.randint(1,100,24).reshape(6,4)
# print(a)
# print("sorting ",np.sort(a,axis=0)[::-1])


# ========================= np.append==================================
# import numpy as np
# a=np.random.randint(1,100,24).reshape(6,4)
# print(a)
# print("sorting ",np.append(a,np.ones((a.shape[0],1)),axis=1))
#


# ========================  np.unique  =============================
#
# import numpy as np
# from numpy.ma.extras import unique
#
# a=np.array([1,2,2,3,4,4,7,6,6,8,9,9])
# print(unique(a))

# ===================  np.where  =====================================
#
# import numpy as np
# a=np.random.randint(1,100,24)
# print(np.where(a>50,0,a))



# =========================  np.histogram ==========================

#
# import numpy as np
# a=np.random.randint(1,100,24)
# print(a)
# print(np.histogram(a,bins=[0,10,20,30,40,50,60,70,80,90,100]))
# print(np.histogram(a,bins=[0,50,100]))


# ================== np.corref  mean corelation coeffiecent  ==========================
#
# import numpy as np
# salary=np.array([20000,30000,40000 ,50000,60000])
# expriences=np.array([1,2,3,2,4])
# print(salary,expriences)
# print(np.corrcoef(salary,expriences))



# =====================================   put  ============================================
# import numpy as np
#
# a = np.random.randint(1, 100, 24)
# print("Original array:", a)
#
# # Modify the first two elements of the array
# print(np.put(a, [0, 1], [110, 530]))
#
# # Print the modified array
# print(a)
#




# ============================   set diff1d ===========================
#
# n=np.array([1,2,3,4])
# m=np.array([2,3,5,7])
# print(np.setdiff1d(n,m))




# ==============================    pandas  ===========================


#
#
# country=["pakistan","india","nepal","srikanka","USA"]
# print(pd.Series(country))


# ====================   list in pandass  =================================
#
import numpy as np
# import pandas as pd
# mark=[58,65,70,80,90]
# subject=["math","english","urdu","punjabi","bio"]
# print(pd.Series(mark, index=subject,name="shoaib ky marks"))




# =============================distionery  in pandas  ==========================
#
# import numpy as np
# import pandas as pd
# marks={"punjabi":95,"math":90,"urdu":80,"english":75}
# print(pd.Series(marks,name="nitish ky marks"))


# ===========================  size of pandas  ========================

#
# import numpy as np
# import pandas as pd
#
# # Create a Pandas Series from the dictionary
# marks = {"punjabi": 95, "math": 90, "urdu": 80, "english": 75}
# marks_series = pd.Series(marks ,name="shoaib ky marks")
#
# # Print the Series
# print(marks_series)
#
# # Print the size of the Series
# print("size of the series", marks_series.size)
#
# print("find the datatype of the series  =",marks_series.dtype)
#
# print(marks_series.name)


#     ==================================   name attribute  ===========================
#
# import pandas as pd
#
# # Create a Pandas Series from the dictionary
# marks = {"punjabi": 95, "math": 90, "urdu": 80, "english": 75}
# marks_series=pd.Series(marks,name="shoaib ky marks")
# print(marks_series)
#
# print(marks_series.name)

# ===========================  is uniques======================


#
# import numpy as np
# import pandas as np
# import pandas as pd
#
# a=[1,2,3,4,5,6,7,7,89]
# b=pd.Series(a)
# print(b)
# print("this series is unique are not= ",b.index)



# ========================  values ============================

#
# import pandas as pd
# marks = {"punjabi": 95, "math": 90, "urdu": 80, "english": 75}
# marks_series=pd.Series(marks)
# print(marks_series.values)


# ==========================    read csv ===============================
# import pandas as pd
#
#
# data = pd.read_csv('subs.csv')
#
# print(data)
# series = data.squeeze()
# print(type(series))
#
#

# =========================   example 2===============
#
# import pandas as pd
# data=pd.read_csv('kohli_ipl.csv',index_col='match_no')
# series=data.squeeze()
# print(series)



import pandas as pd
# data=pd.read_csv('kohli_ipl.csv',index_col='match_no')
# print(data)
#
# print((data.head()))
# print(data.head(10))


#
# print("value count = ", data.sort_values(by='runs'))

# print("value count = ", data.sort_values(by='runs', ascending=False))


#
# import pandas as pd
#
# data = pd.read_csv('subs.csv')
# print(data)
# print(data.describe())



#  ===============================     Series of indexing=============================
#
# import pandas as pd
# data=pd.read_csv('kohli_ipl.csv',index_col='match_no')
# print(data.squeeze())
#
# #print(data[[1,3,5,7]])
# print(data.loc[[1, 3, 5, 7]])


# =================  editng  or writing  indexing ==================
#
# import pandas as pd
#
# marks = {"punjabi": 95, "math": 90, "urdu": 80, "english": 75}
# marks_series = pd.Series(marks)
#
#
# marks_series.iloc[1] =100
#
# marks_series['pak_study']=97
#
# print(marks_series)


# ==============================  writing and editing  slices ================

# import pandas as pd
# x=pd.Series([3,45,67,8,9,78,99,100])
# x[2:5]=[70,71,72]
# print(x)


# ======================   fency indexing  ====================================
#
# import pandas as pd
# x=pd.Series([3,45,67,8,9,78,99,100])
# x[[0,2,4]]=[0,0,0]
# print(x)


# import pandas as pd
# data=pd.read_csv('subs.csv')
# data_series=data.squeeze()
# print(len(data_series))
# print(type(data_series))
# print(dir(data_series))


# =========================   conversion ======================

#
# import pandas as pd
#
# marks = {"punjabi": 95, "math": 90, "urdu": 80, "english": 75}
# marks_series = pd.Series(marks)
# print(dict(marks_series))




  # =========================    membership operator  or in operator  ========================
#
# import pandas as pd
# data = pd.read_csv('subs.csv')
# data_series=data.squeeze()
# # print(data_series)
# # print(155 in data_series.values)
#
# for i in data_series:
#     print(i)


# ====================  arithemic operatioon================================

#
# import pandas as pd
# marks = {"punjabi": 95, "math": 90, "urdu": 80, "english": 75}
# marks_series = pd.Series(marks)
# print(100-marks_series)



# =======================    relation operator   ============================
# import matplotlib.pyplot  as plt
# import pandas as pd
# data=pd.read_csv('kohli_ipl.csv',index_col='match_no')
# data_series=data.squeeze()
# print(data_series[data_series>=50].size)
# data_series.plot(kind='pie')
# plt.show()




# ===============================   how to make data frame  =================================

# import numpy as np
# import pandas as pd
# student_data=[
#     [80,70, 7],
#     [90,80,8],
#     [120,100,10],
#     [70,60,6]
# ]
# df = pd.DataFrame(student_data, columns=["iq", "marks", "package"])
# print(df)


# ========================  other method to create data frame===================
# import numpy as np
# import pandas as pd
# student_data={
#     "iq":[80,90,120,70],
#     "marks":[70,80,100,60],
#     "package":[7,8,10,6]
# }
# pf=pd.DataFrame(student_data)
# print(pf)


# =============================
# import pandas as pd
# pf=pd.read_csv('moviesindian.csv')
# pff=pd.read_csv('ipl-matches.csv')
#
# student_data={
#     "name":["ali","shoaib","abdullah","asad","humza","abuhararah"],
#     "iq":[80,90,120,70,0,0],
#     "marks":[70,80,100,60,0,0],
#     "package":[7,8,10,6,0,0]
# }
# pfff=pd.DataFrame(student_data)


# print(pf.shape)
# print(pff.shape)


# ======================  dtype  ============
# print(pf.dtypes)
# print(pff.dtypes)


 # ===================  index  ===========

# print(pf.index)

# ========================  colum  ====================
#
# print(pf.columns)


# ===================  value===========================

# print(pfff.values)


# =====================  head and tail ===============

# print(pff.sample(5))



# ===========================  info  ======================

# print(pff.info())


# =====================  discribe  ========================

# print(pff.describe())

# ======================  is null()  =====================

# print(pf.isnull().sum())

#   ====================   duplicated ================

# print(pf.duplicated().sum())

# ===================  rename  =====================

# print(pfff.rename(columns={'marks':'persentag','package':'salay'}))


# ===========================  math operatiion  ======================

# print(pfff)
# print(pfff.sum(axis=0))

# =================  mean =====================================
#
# print(pfff)
# print(pfff.mean())


# ===================  min  =======================
#
# print(pfff)
# print(pfff.min())

# =====================  max =======================
#
# print(pfff)
# print(pfff.max())

#
# # ========================  single columns   fetch ======================
# print(pf)
# print(pf['title_x'])


#
# ========================  multiply columns   fetch ======================
# print(pf)
# print(pf[['title_x', 'release_date', 'actors']])


# Step 1: Check available columns
# print(pff.columns)
#
# # Step 2: Once you confirm the exact names, adjust accordingly:
# print(pff[['Team1', 'Team2', 'WinningTeam']])  # Make sure the names are exactly as shown in pff.columns


# =======================  fetch row from in data frame  ======================
# pfff.set_index('name',inplace=True)
# print(pfff)
# print(pfff.iloc[0])


# #   ===============  iloc  ==========================
# print(pf.iloc[[0,5,7]])


# ================loc=============================
# print(pfff)
# pfff.set_index('name',inplace=True)
# # print(pfff)
# print(pfff.loc['shoaib':'humza':2])

# ======================== select both row and colum  =====================
#
# print(pf)
#
# print(pf.iloc[0:3,0:3])


# ================================  data filtering  ================================
#
# Finaal=pff['MatchNumber']=='Final'
# fd_new=pff[Finaal]
# print(fd_new[['Season','WinningTeam']])


# =============================  super over  ==========================
#
# sp=pff[pff['SuperOver']=='Y'].shape
# print(sp)


# ================     how many channai super king won match in kolkatah  ================
#
# csp = pff[(pff['City'] == 'Kolkata') & (pff['WinningTeam'] == 'Chennai Super Kings')]
# cspp = pff[(pff['City'] == 'Kolkata') & (pff['WinningTeam'] == 'Chennai Super Kings')].shape[0]
#
# print(csp)
# print("Won the total ",cspp)


# ===================  toss winer has winer match  ===========================
#
# print(pff.columns)
#
# match=(pff[pff['TossWinner']==pff['WinningTeam']].shape[0]/pff.shape[0])*100
# print(match)


# =======================  find these movies how rating is 8 and voting is graeter 10000  ================
#
# pratin=pf[(pf['imdb_rating']>8) & (pf['imdb_votes']>10000)]
# pratinn=pf[(pf['imdb_rating']>8) & (pf['imdb_votes']>10000)].shape[0]
# print(pratin)
# print(pratinn)


# ====================  rating the action heigher then 7.5  =====================================
# Step 1: Create boolean masks (True/False values for each row)
# mask1 = pf['genres'].str.split('|').apply(lambda x: 'Action' in x)
# mask1 = pf['genres'].str.contains("Action")
# mask2 = pf['imdb_rating'] > 7.5
#
# # Step 2: Combine masks
# final_mask = mask1 & mask2
#
# # Step 3: Filter and print matching rows
# print(pf[final_mask].shape[0])


# ==============  how to add new colum in data frame  ===============================
#
# print(pf)
#
# colum=pf['country']='pakistan'
# print(colum)
#
# print(pf.columns)


# ==========================   astype  ====================

# print(pff.info())
#
# pff['ID']=pff['ID'].astype('int32')
# print(pff.info())
#



# ===================  drop duplication  ===========================]
# import numpy as np
# import pandas as pd
# pf=pd.Series([1,1,2,3,4,4,5,5,np.nan,np.nan,6,7,np.nan,8,9,9])
# print(pf.drop_duplicates(keep="last"))
#
# dupl=pf[pf.duplicated()].size
# print(dupl)

# print(pf.isnull().sum())


# ===================  dropna =================================

# print(pf.dropna().size)

# ========================  fillna  ===========================

# print(pf.fillna(1))








# ========================  isin ===============================
#
# import pandas as pd
# pf=pd.read_csv('kohli_ipl.csv')
# print(pf[pf.isin([49,99])])

#


# ===================================apply  ====================================
#
# import pandas as pd
# pf =pd.read_csv('moviesindian.csv')
# print(pf.apply(lambda x: str(x).split()[0]).upper())



# ========================  copy  =====================================
# import pandas as pd
# pf=pd.read_csv('kohli_ipl.csv')
# print(pf)
#



# ===========================  count_value   find out frequency of each  item  ======================
# import pandas as pd
#
# student_data = [
#     [100, 90, 9],
#     [120, 100, 10],
#     [90, 80, 8],
#     [90, 80, 8],
#     [90, 80, 8]  # ← valid row, not a variable assignment
# ]
#
# pf = pd.DataFrame(student_data, columns=["iq", "marks", "package"])
# print(pf.value_counts())



# =========================  value count  ============================
import matplotlib.pyplot as plt
# import pandas as pd
# pf=pd.read_csv('ipl-matches.csv')


# print(pf[~pf['MatchNumber'].str.isdigit()]['Player_of_Match'].value_counts())

#
# print(pf['TossDecision'].value_counts().plot(kind='pie'))
# plt.show()
#

#
# print(pf['Team1'].value_counts() + pf['Team1'].value_counts())

# ============================   sort_values   ==========================
#
# import pandas as pd
# pf=pd.Series([1,2,34,56,13,45,23,78,90,5,2])
# print(pf.sort_values(ascending=False))



# ============================   sort_values  in dataframe   ==========================
#
# import pandas as pd
# pf=pd.read_csv('moviesindian.csv')
#
# print(pf.sort_values('title_x', ascending=False))



# =========================   rank  =========================
#
# import pandas as pd
# pf=pd.read_csv('batsman_runs_ipl.csv')
#
# pf['batsmann_rank'] = pf['batsman_run'].rank(ascending=False)
#
# print(pf.sort_values('batsman_run',ascending=False))



# ================================  sort index  =======================

import pandas as pd
# marks={
#     'math':70,
#     'english':90,
#     'urdu':75,
#     'punjabi':85
# }
# mark_series=pd.Series(marks)
# print(mark_series.sort_index())
#
# pf=pd.read_csv('moviesindian.csv')
# print(pf.sort_index(ascending=False))



# =======================  set indexing  =====================================
#
# import pandas as pd
# pf=pd.read_csv('batsman_runs_ipl.csv')
# pf['batsman_rank']=pf['batsman_run'].rank(ascending=False)
#
#
# batsman=pf.set_index('batter')
# print(batsman.reset_index().set_index('batsman_rank'))


# =====================  rename  =====================
#
# import pandas as pd
# pf=pd.read_csv('moviesindian.csv')
#
# pf.set_index('title_x',inplace=True)

# pf.rename(columns={'imdb_id': 'imbd', 'poster_path': 'link'}, inplace=True)
#
# pf.rename(index={'Uri: The Surgical Strike': 'uri', 'Battalion 609': 'Battalion'},inplace=True)
# print(pf)
# print(pf.columns)



# =======================  unique ==========================

# import pandas as pd
# temp=pd.Series([1,2,2,3,4,4,5,6,6,7])
# print(temp.unique())


#   =========================  nunique ==========================


# import pandas as pd
# temp=pd.read_csv('ipl-matches.csv')
# print(temp['Season'].unique())



# =========================  drop duplicate  =============================
#
# import pandas as pd
#
# student_data = [
#     [100, 90, 9],
#     [120, 100, 10],
#     [90, 80, 8],
#     [90, 80, 8],
#     [90, 80, 8]  # ← valid row, not a variable assignment
# ]
#
# pf = pd.DataFrame(student_data, columns=["iq", "marks", "package"])
# print(pf.drop_duplicates(keep="first"))

# ==========  how the last match varat kolhi play in delhi  ============================
#
# import pandas as pd
# pf=pd.read_csv('ipl-matches.csv')
# # print(pf.info())
# pf['all_layer']=pf['Team1Players'] + pf['Team2Players']
#
# def did_kolhi_play(play_list):
#     return 'V Kolhi' in play_list
# pf['did_kolhi_play']=pf['all_layer'].apply(did_kolhi_play)
# pf[(pf['City'] == 'Delhi') & (pf['did_kolhi_play'] == True)].drop_duplicates(subset=['City', 'did_kolhi_play'], keep='first')
#
# print(pf)




# ===========================  drop  ============================================


#
#
# import pandas as pd
# temp=pd.Series([1,2,2,3,4,4,5,6,6,7])
# print(temp)
# print(temp.drop(index=[0,9]))


#====================================   apply on data_Frame  ==========================
#
# import pandas as pd
# pf=[
#     [(2,3),(4,3)],
#     [(1,3),(2,5)],
#     [(4,5),(4,7)],
#     [(7, 9), (6,1)],
#     [(7, 9), (11, 17)],
#
#
# ]
# pff=pd.DataFrame(pf,columns=(['point x','point y']))
# print(pff)
#
#
# def eudlidean(row):
#     row_x=row['point x']
#     row_y=row['point y']
#     return ((row_x[0]-row_y[1])**2 + (row_x[1] - row_y[1])**2)**0.5
#
# print(pff.apply(eudlidean,axis=1))



# ===============================   groupby ======================
# import numpy as np
# import pandas as pd
# pf=pd.read_csv('imdb-top-1000.csv')
# # print(pf.info())
# genres=pf.groupby('Genre')
# # print(genres.min())
# print(genres['IMDB_Rating'].mean())




# ==============  find the top 3 genres by total earning ==========================================================
#
# import pandas as pd
# pf=pd.read_csv('imdb-top-1000.csv')
# # print(pf.info())
# genres=pf.groupby('Genre')
# print(genres.sum()['Gross'].sort_values(ascending=False).head(3))



#=============================   find the genres  with highest averages  IMDB rating=======================
#
# import pandas as pd
# pf=pd.read_csv('imdb-top-1000.csv')
# # print(pf.info())
#
# genres=pf.groupby('Genre')
# print(genres['IMDB_Rating'].mean().sort_values(ascending=False).head(1))





# ========================   find director  with most popularity  ====================================

# import pandas as pd
# pf=pd.read_csv('imdb-top-1000.csv')
# # print(pf.info())
# print(pf.groupby('Director')['No_of_Votes'].sum().sort_values(ascending=False).head(1))


# ========================  find the highest rating of each jhonar  ====================
#
# import pandas as pd
# pf=pd.read_csv('imdb-top-1000.csv')
# # print(pf.info())
# print(pf.groupby('Genre')['IMDB_Rating'].max().sort_values(ascending=False).head(1))



# ========================  find the number of movies of each actor   ==============================

#
# import pandas as pd
#
# pf=pd.read_csv('imdb-top-1000.csv')
# print(pf.info())
# star=pf['Star1'].value_counts().max()
# print(pf.groupby('Star1')['Released_Year'].count().sort_values(ascending=False).head(1))

# ========================= find  the row of group  ==================================

# grnes=pf.groupby('Genre')
# print(grnes.size())


# =========================  find the first  movies of each group  ===========================
#
# grnes=pf.groupby('Genre')
# print(grnes.first())

# ======================   if you want seventh movies of each group  ============================

#
# grnes=pf.groupby('Genre')
# print(grnes.nth(7))



# =======================   you can see all the movies of particular group   you can use the get_ groups attributes ==
#
# grnes=pf.groupby('Genre')
# # print(grnes.size())
# print(grnes.get_group('Comedy'))



# ========================================   group ==========================================

#
#
# import pandas as pd
# pf=pd.read_csv('imdb-top-1000.csv')
# grenes=pf.groupby('Genre')
# print(grenes.groups)

# =====================================   discribes  ===================================

# import pandas as pd
# pf=pd.read_csv('imdb-top-1000.csv')
# grenes=pf.groupby('Genre')
# print(grenes.describe())

# =================================  sample  ====================================
#
# import pandas as pd
# pf=pd.read_csv('imdb-top-1000.csv')
# grenes=pf.groupby('Genre')
# print(grenes.sample(3,replace=True))



# =====================  aggricate function ===============================



# import pandas as pd
# pf=pd.read_csv('imdb-top-1000.csv')
# grenes=pf.groupby('Genre')
# print(grenes.agg(
#     {
#         'Runtime':'mean',
#         'IMDB_Rating':'sum',
#         'No_of_Votes':'sum',
#         'Gross':'sum'
#     }
# ))

# ===============================   if you want three operation on each columns  throught aggrication  function =======
#
# import pandas as pd
# pf=pd.read_csv('imdb-top-1000.csv')
# grenes=pf.groupby('Genre')
# print(grenes.agg(['min','max','mean']))


#
# import pandas as pd
#
# # Read the CSV file
# pf = pd.read_csv('imdb-top-1000.csv')
#
# # Group by 'Genre'
# genres = pf.groupby('Genre')
#
# # Apply aggregation only on numeric columns
# numeric_columns = pf.select_dtypes(include='number').columns
# print(genres[numeric_columns].agg(['min', 'max', 'mean']))



# =========================  looping on group  ====================================

# import pandas as pd
# pf=pd.read_csv('imdb-top-1000.csv')
# # print(pf.info())
# genre=pf.groupby('Genre')
#
# for group,data in genre:
#  print(data[data['IMDB_Rating']==data['IMDB_Rating'].max()])
#
#
#



# ===============================   split apply combine  ==========================

# ==================  find account the number of the movies  start with  A   =======================

#
# import pandas as pd
# pf=pd.read_csv('imdb-top-1000.csv')
# # print(pf.info())
# genre=pf.groupby('Genre')
#
# def foo(group):
#  return group['Series_Title'].str.startswith('A').sum()
#
#
# print(genre.apply(foo))


# ==================  find the ranking of each movies in the group  according to  IMBD sorces  ===========

#
# import pandas as pd
# pf=pd.read_csv('imdb-top-1000.csv')
# # print(pf.info())
# genre=pf.groupby('Genre')
#
# def foo(group):
#  group['ranking_movies']=group['IMDB_Rating'].rank(ascending=False)
#  return group
#
#
# print(genre.apply(foo))





# ==========================  find normalized  IMDB  rating  group wise   ==================

#
# import pandas as pd
# pf=pd.read_csv('imdb-top-1000.csv')
# # print(pf.info())
# genre=pf.groupby('Genre')
#
# def normalization(group):
#  group['normalized_rating']=group['IMDB_Rating']-group['IMDB_Rating'].min()/group['IMDB_Rating'].max()-group['IMDB_Rating'].min()
#  return group
#
#
# print(genre.apply(normalization))


# ==============================   groupby on multiply columns  ========================


# import pandas as pd
# pf=pd.read_csv('imdb-top-1000.csv')
# # print(pf.info())
# genre=pf.groupby(['Director','Star1'])
# print(genre.get_group(('Aamir Khan','Amole Gupte')))


# =======================  why director and the actor most earning in the movies ====================
#
# import pandas as pd
# pf=pd.read_csv('imdb-top-1000.csv')
# # print(pf.info())
# genre=pf.groupby(['Director','Star1'])
#
# print(genre['Gross'].sum().sort_values(ascending=False))



# =========================   find the best  matascorce  averages in actor and genre combination  ============

#
# import pandas as pd
# pf=pd.read_csv('imdb-top-1000.csv')
# # print(pf.info())
# genre=pf.groupby(['Director ','Star1'])
#
#
#  print(genre.get_group(['Star1','Genre']))

# ==========================  aggrigate function on multiply groupby   =================
# import pandas as pd
# 
# # Read CSV file
# pf = pd.read_csv('imdb-top-1000.csv')
# 
# # Select only numeric columns you want to aggregate
# numeric_cols = ['IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Gross']  # Adjust based on your dataset
# 
# # Group by Director and Star1, then aggregate numeric columns
# genre = pf.groupby(['Director', 'Star1'])[numeric_cols]
# 
# # Now apply aggregation
# print(genre.agg(['min', 'max', 'mean']))

