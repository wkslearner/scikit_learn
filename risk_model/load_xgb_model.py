
import pickle
import numpy as np
import pandas as pd
import argparse

# 载入模型
xgb_model=pickle.load(open('xgb_model.pkl','rb'))

# 字段参数获取
feature_list=['S24XSLJSQDDS_001','SFZDZSZSF_002','DEPOSIT_001','DQDDZCCJSJC_001',
'DQDDCJYPGMYSJC_001','PGSHDZ_001','TD90TNSFZGLBTFYXJRPTS_001','TD90TSFZGLDSFFWPTS_001']
type_list=[float,str,float,float,float,str,float,float]
parse=argparse.ArgumentParser()
for feature,feature_type in zip(feature_list,type_list):
    parse.add_argument(feature,type=feature_type)
args=parse.parse_args()

# 分类特征编码函数
def data_turn(value,turn_dict):
    equal_num=0
    repalce_value=''
    for item in turn_dict.items():
        if value==item[1]:
            repalce_value=item[0]
            equal_num+=1
    if equal_num>=2:
        raise ValueError(f'thire are too many value equal to {value}')
    if repalce_value=='':
        raise ValueError(f'feature is null please check it ')

    return repalce_value

# 概率转换为分数
def prob_to_score(prob, basePoint, PDO):
    B=PDO/np.log(2)
    A=basePoint+B*np.log(1/20)  #设定基准分的odds为1/20
    y = np.log(prob/(1-prob))  #log(odds)

    return  int(A-B*y)

def collage_transform(data):
    if type(data)!=str:
        raise TypeError(f'the type of parameter data must be string')
    if '大学' in data or '学院' in data:
        return 'Y'
    else:
        return 'N'

# 分类特征编码
sf_dict=xgb_model.cate_dict['identity_province']
print(sf_dict)
collage_dict=xgb_model.cate_dict['college_tf']
province=data_turn(args.SFZDZSZSF_002,sf_dict)
collage_tf=data_turn(collage_transform(args.PGSHDZ_001),collage_dict)

# 入参字段（固定位置）
item=[args.DEPOSIT_001,args.DQDDZCCJSJC_001,province,collage_tf,args.DQDDCJYPGMYSJC_001,
      args.S24XSLJSQDDS_001,args.TD90TSFZGLDSFFWPTS_001,args.TD90TNSFZGLBTFYXJRPTS_001]

# 模型预测
res_prob=xgb_model.predict_proba(np.array(item).reshape(1,-1))[:,1][0]

# 模型分数
score=prob_to_score(res_prob,600,20)
print(score)


