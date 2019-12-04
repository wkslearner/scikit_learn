##导入包
from sklearn.externals import joblib
import pandas as pd



'''加载模型文件'''
classifier=joblib.load('/Users/admin/Downloads/model_xgboost_20191203.pkl')

'''获得特征名称'''
feature_name=['x_lyw_long_rent_apply_behavior_history_cash_max_differ_min_include_sameday',
       'x_lyw_long_rent_apply_behavior_history_long_rent_number_within0_365days',
       'x_lyw_long_rent_apply_behavior_history_long_rent_number_within0_90days',
       'x_lyw_long_rent_apply_behavior_history_cash_max_differ_min',
       'x_lyw_long_rent_apply_behavior_history_long_rent_number_within0_30days_ratio_by_365days',
       'x_lyw_long_rent_apply_behavior_history_long_rent_number_within0_30days',
       'x_lyw_long_rent_apply_behavior_history_long_rent_number_within0_180days']

'''获得特征名称对应特征值'''
feature_name_value=[0.2917457981951744,0.1767576429319638,-0.2437415639581259,0.2477562726255736,-0.2897342924729717,-0.2152604844266684,-0.03764115967106792]

'''特征名称及值对应'''
df_matrix=pd.DataFrame(feature_name_value).T
df_matrix.columns=feature_name

'''预测概率计算'''
result_predict=classifier.predict_proba(df_matrix)[:,1]

'''返回结果'''
result=result_predict[0]

print('预测概率结果：'+str(result))


