# Financial-fraud-prediction<br>
1、导入数据，并查看数据信息
----------
```
# 导入数据文件
df = pd.read_csv('F:\Python\data\LoanStats_2016Q3.csv', skiprows=1, low_memory=False)
df.info() # 查看文件属性
```
2、对数据进行预处理，将相关度不大的属性列删除<br>
----------
```
# drop id and member_id
df.drop('id',1,inplace=True)
df.drop('member_id',1,inplace=True)

# drop nan rows
df.dropna(axis=0,how='all',inplace=True)
df.info()

# emp_title is too many to ignore
df.drop('emp_title',1,inplace=True)
```

3、查看数据类别，将缺失值较大的属性列删除<br>
----------
```
# drop hige missing_pct 
df.drop('desc',1,inplace=True)
df.drop('verification_status_joint',1,inplace=True)

# drop unnecessary object
df.drop('term',1,inplace=True)
df.drop('issue_d',1,inplace=True)
df.drop('purpose',1,inplace=True)
df.drop('title',1,inplace=True)
df.drop('zip_code',1,inplace=True)
df.drop('addr_state',1,inplace=True)
df.drop('earliest_cr_line',1,inplace=True)
df.drop('revol_util',1,inplace=True)

# drop after loan
df.drop(['out_prncp','out_prncp_inv','total_pymnt','total_pymnt_inv','total_rec_prncp', 'grade', 'sub_grade'] ,1, inplace=True)
df.drop(['total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee','collection_recovery_fee'],1, inplace=True)
df.drop(['last_pymnt_d','last_pymnt_amnt','next_pymnt_d','last_credit_pull_d'],1, inplace=True)
df.drop(['policy_code'],1, inplace=True)
```

4.整理数据列，关注loan_status属性列，并计算与该属性列相关系数，相关系数高的可以删除<br>
------------
```
# focus on loan_status
df.loan_status.value_counts()

# Current and Fully Paid regard as excellent 1, late regard as bad loan 0, others regard ass nan
df.loan_status.replace('Current',int(1),inplace=True)
df.loan_status.replace('Fully Paid',int(1),inplace=True)
df.loan_status.replace('Charged Off',np.nan,inplace=True)
df.loan_status.replace('In Grace Period',np.nan,inplace=True)
df.loan_status.replace('Default',np.nan,inplace=True)
df.loan_status.replace('Late (31-120 days)',int(0),inplace=True)
df.loan_status.replace('Late (16-30 days)',int(0),inplace=True)
```
```
# find highly corr Data
cor = df.corr()
cor.loc[:,:] = np.tril(cor,k=-1)
cor = cor.stack()
cor[abs(cor) > 0.5]

# drop the columns highly correlated with loan_status
df.drop(['funded_amnt','funded_amnt_inv','installment'],1,inplace=True)
```
5.对已处理好的数据进行建模训练，测试集为0.2<br>
----------
```
Y = df.loan_status
df.drop('loan_status',1,inplace=True)
X = df

# train_data and test_data
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=1)
```
6.使用GBRT对数据进行建模训练<br>
---------
```
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
from sklearn.preprocessing import OneHotEncoder
```
```
param_grid = {'learning_rate': [0.1],
              'max_depth': [2],
              'min_samples_split': [50,100],
              'n_estimators': [100,200]
              }


est = GridSearchCV(ensemble.GradientBoostingRegressor(),
                   param_grid, n_jobs=4, refit=True)

est.fit(x_train, y_train)

best_params = est.best_params_
print(best_params)
```
```
%%time
est = ensemble.GradientBoostingRegressor(min_samples_split=50,n_estimators=300,
                                        learning_rate=0.1,max_depth=1,random_state=0,
                                        loss='ls').fit(x_train,y_train)
est.score(x_test,y_test)                                  
```
7、测试模型，找出10个最重要属性<br>
-------
```
def computer_ks(data):
    sorted_list = data.sort_values(['predict'],ascending=True)
    total_bad = sorted_list['label'].sum(axis=None,skipna=None,level=None,numeric_only=None)/3
    total_good = sorted_list.shape[0] - total_bad
    
    max_ks = 0.0
    good_count = 0.0
    bad_count = 0.0
    for index, row in sorted_list.iterrows():
        if row['label'] == 3:
            bad_count += 1.0
        else:
            good_count += 1.0
            
        val = bad_count/total_bad - good_count/total_good
        max_ks = max(max_ks,val)
        
    return max_ks
```

```
feature_importance = est.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())

indices = np.argsort(feature_importance)[-10:]
plt.barh(np.arange(10),feature_importance[indices],color='green',alpha=0.4)
plt.yticks(np.arange(10+0.25),np.array(X.columns)[indices])
_ = plt.xlabel('Relative importance'), plt.title('Top Ten Important Variables')
```
8、最终结果<br>
-------


