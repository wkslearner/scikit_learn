
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve,auc,roc_auc_score
import matplotlib.pyplot as plt

### Calculate the KS and AR for the socrecard model
def KS_AR(df, score, target):
    '''
    :param df: the dataset containing probability and bad indicator
    :param score:
    :param target:
    :return:
    '''

    total = df.groupby([score])[target].count()
    bad = df.groupby([score])[target].sum()
    all = pd.DataFrame({'total':total, 'bad':bad})
    all['good'] = all['total'] - all['bad']
    all[score] = all.index
    all = all.sort_values(by=score,ascending=False)
    all.index = range(len(all))
    all['badCumRate'] = all['bad'].cumsum() / all['bad'].sum()
    all['goodCumRate'] = all['good'].cumsum() / all['good'].sum()
    all['totalPcnt'] = all['total'] / all['total'].sum()
    arList = [0.5 * all.loc[0, 'badCumRate'] * all.loc[0, 'totalPcnt']]
    for j in range(1, len(all)):
        ar0 = 0.5 * sum(all.loc[j - 1:j, 'badCumRate']) * all.loc[j, 'totalPcnt']
        arList.append(ar0)
    arIndex = (2 * sum(arList) - 1) / (all['good'].sum() * 1.0 / all['total'].sum())
    KS = all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
    return {'AR':arIndex, 'KS': max(KS)}



def Prob2Score(prob, basePoint, PDO):
    #将概率转化成分数
    y = np.log(prob/(1-prob))
    return  int(basePoint+PDO/np.log(2)*(-y))



'''PSI值计算'''
def PSI(dataframe_act, dataframe_pre, actual_p, predit_p):
    '''
    :param dataframe_act: 训练集数据框
    :param dataframe_pre: 测试集数据框
    :param actual_p: 训练集样本预测概率，放在训练集数据框中
    :param predit_p: 测试集样本预测概率，放在测试集数据框中
    :return: psi 最终结果值
    '''
    total_act = dataframe_act.shape[0]
    total_pre = dataframe_pre.shape[0]

    psi = 0
    for i in range(9):
        act = dataframe_act[(dataframe_act[actual_p] >= 0.1 * i) & (dataframe_act[actual_p] < 0.1 * (i + 1))][
            actual_p].count()
        pre = dataframe_pre[(dataframe_pre[predit_p] >= 0.1 * i) & (dataframe_pre[predit_p] < 0.1 * (i + 1))][
            predit_p].count()
        proportion_act = act / total_act
        proportion_pre = pre / total_pre
        #print(proportion_pre,proportion_act)
        if proportion_pre==0:
            psi=psi+0
        else:
            psi = psi + (proportion_act - proportion_pre) * np.log2(proportion_act / proportion_pre)

    act = dataframe_act[(dataframe_act[actual_p] >= 0.9)][actual_p].count()
    pre = dataframe_pre[(dataframe_pre[predit_p] >= 0.9)][predit_p].count()
    proportion_act = act / total_act
    proportion_pre = pre / total_pre

    if proportion_pre == 0:
        psi = psi + 0
    else:
        psi = psi + (proportion_act - proportion_pre) * np.log2(proportion_act / proportion_pre)

    return psi



'''
群体稳定性指标(population stability index),
公式： psi = sum(（实际占比-预期占比）*ln(实际占比/预期占比))
举个例子解释下，比如训练一个logistic回归模型，预测时候会有个概率输出p。你测试集上的输出设定为p1吧，
将它从小到大排序后10等分，如0-0.1,0.1-0.2,......。
现在你用这个模型去对新的样本进行预测，预测结果叫p2,按p1的区间也划分为10等分。
实际占比就是p2上在各区间的用户占比，预期占比就是p1上各区间的用户占比。
意义就是如果模型跟稳定，那么p1和p2上各区间的用户应该是相近的，占比不会变动很大，也就是预测出来的概率不会差距很大。
一般认为psi小于0.1时候模型稳定性很高，0.1-0.25一般，大于0.25模型稳定性差，建议重做。
'''


def ROC_curve(real_value,pred_value,title='Receiver Operating Characteristic',legend_position='lower right',
              ylabel='True Positive Rate',xlabel='False Positive Rate'):

    fpr, tpr, threshold = roc_curve(real_value,pred_value)
    roc_auc = auc(fpr, tpr)

    plt.title(title)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc=legend_position)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()
