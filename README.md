# Readme.md
### 天池精准医疗大赛-糖尿病遗传风险预测
##### Top12 思路 由于初赛和复赛题目相差太大，谨在此给出复赛的一点思路权当抛砖引玉

#### 特征工程
##### 新特征构造
######  1.构造加减乘除四则运算特征，做特征间的交互
######  2.构造特征本身的乘方，幂方，开方等数值特征
######  3.利用多项式特征包来构造特征(线上表现不行)
  
##### 缺失值的处理
######  1.观察数据分布，对于缺失数据在非长尾的特征，均值填充/中值填充
######  2.把缺失值的特征当Label，考虑Label Propagation传播算法，半监督填充Label
######  3.不用GBDT等模型填充的原因是对于缺失值较多的(40%-75%)，无法保证数据的分布一致
######  4.将缺失值数量超过75%的进行删除
  
##### 模型的选择
###### 其实可以很轻松的发现这题数据量小，利用堆叠复杂的模型可能导致过拟合，故我们采用的是贪心法选择最优特征，基本框架为
```
if Choose_Best_Feature(now_feature)<the_last_best:
    now_feature.pop()
else:
    print('Now CV:',cv_mean)
```
###### 在Choose_Best_Feature模块中，是每次加入一个新特征计算的整体CV的值，不断更新最优值，显然，其一，这种选择方法是具有一定的盲目性的，贪心法陷入的是局部最优解，可能该组特征向量只是近似最优解，故可以考虑引入模拟退火机制，Random一个数满足某个条件则改变最优值；其二，如果数据量大，特征多，在时间效率上是无法承受的，故笔者提出了一种小技巧仅供参考，小技巧有两个方向
```
def get_pic(model,feature_name):
    ans = DF()
    ans['name'] = feature_name
    ans['score'] = model.feature_importances_
    print(ans[ans['score']>0].shape)
    return ans.sort_values(by=['score'],ascending=False).reset_index(drop=True)
    
nums = 45
feature_name1 = train_data[feature_name].columns
get_ans_face = list(set(get_pic(lgb_model,feature_name1).head(nums)['name'])|set(get_pic(xgb_model,feature_name1).head(nums)['name'])|set(get_pic(gbc_model,feature_name1).head(nums)['name']))
# get_ans_face = list(set(get_pic(lgb_model,feature_name1).head(nums)['name'])&set(get_pic(xgb_model,feature_name1).head(nums)['name'])&set(get_pic(gbc_model,feature_name1).head(nums)['name']))
# 先训练好三个模型 第一种方法是将三个模型的Feature_importances的Top K选择出来后，将这些特征取并集；而第二种方法则是取交集
```
###### 在经验上 第一种方法所需要设置的nums较小，而第二种方法所需要设置的nums较大，籍此选出较强的特征后进入前文所述的贪心选择法中，即选择出较优的特征向量组，而在Choose_Best_Feature中，笔者使用的是`Xgboost`,`Lightgbm`,`GBDT`三种模型的CV值的平均值量度加入New_Feature对模型的影响，如此可以保证线上与线下的`同增同减`

```
def get_model(nums,cv_fold):
    feature_name1 = train_data[feature_name].columns
    get_ans_face = list(set(get_pic(gbc_model,feature_name1).head(nums)['name'])&set(get_pic(xgb_model,feature_name1).head(nums)['name'])&set(get_pic(lgb_model,feature_name1).head(nums)['name']))
    print('New Feature: ',len(get_ans_face))
    new_lgb_model = lgb.LGBMClassifier(objective='binary',n_estimators=300,max_depth=3,min_child_samples=6,learning_rate=0.102,random_state=1)
    cv_model = cv(new_lgb_model, train_data[get_ans_face], train_label,  cv=cv_fold, scoring='f1')
    new_lgb_model.fit(train_data[get_ans_face], train_label)
    m1 = cv_model.mean()

    new_xgb_model1 = xgb.XGBClassifier(objective='binary:logistic',n_estimators=300,max_depth=4,learning_rate=0.101,random_state=1)
    cv_model = cv(new_xgb_model1, train_data[get_ans_face].values, train_label,  cv=cv_fold, scoring='f1')
    new_xgb_model1.fit(train_data[get_ans_face].values, train_label)
    m2 = cv_model.mean()

    new_gbc_model = GBC(n_estimators=310,subsample=1,min_samples_split=2,max_depth=3,learning_rate=0.1900,min_weight_fraction_leaf=0.1)
    kkk = train_data[get_ans_face].fillna(7)
    cv_model = cv(new_gbc_model, kkk[get_ans_face], train_label,  cv=cv_fold, scoring='f1')
    new_gbc_model.fit(kkk.fillna(7),train_label)

    m3 = cv_model.mean()
    print((m1+m2+m3)/3)
    pro1 = new_lgb_model.predict_proba(test_data[get_ans_face])
    pro2 = new_xgb_model1.predict_proba(test_data[get_ans_face].values)
    pro3 = new_gbc_model.predict_proba(test_data[get_ans_face].fillna(7).values)
    ans = (pro1+pro2+pro3)/3
    return ans
```

###### 在最后的结果提交环节中，也有一个可以参考的小技巧，将选择出来的特征向量组放入三个树模型中可以得到Ans1,Ans2,Ans3,也可以得到概率P1,P2,P3,那么将Ans1、2、3做结果的投票融合得到Ans4，将P1/P2/P3做概率融合得到Ans5,再利用线下表现较好的线性模型利用特征向量组产生Ans6,把Ans4,Ans5,Ans6再进行结果投票即可得到Ans7,Ans7的效果经过笔者的实践证明还不错


