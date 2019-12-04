# %matplotlib inline
from score_card_version1.result_check import calculate_psi,PSI
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("white")
rs = np.random.RandomState(5)

#从整体分布中随机抽取样本
initial = rs.normal(size = 100)
#扩展样本
new = rs.normal(loc = 0.2, size = 120)
print(new)


plot = sns.kdeplot(initial, shade=True)
plot = sns.kdeplot(new, shade=True)
plot.set(yticklabels=[], xticklabels = [])
sns.despine(left=True)
# plt.show()


def scale_range (input, min, max):
    input +=-(np.min(input))
    input /= np.max(input) / (max - min)
    input += min

    return input

buckets = 10
raw_breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
#计算每个分割点所在位置
breakpoints = scale_range(raw_breakpoints, np.min(initial), np.max(initial))

#查看两个数据的分布情况
initial_counts = np.histogram(initial, breakpoints)[0]
new_counts = np.histogram(new, breakpoints)[0]

df = pd.DataFrame({'Bucket': np.arange(1, 11), 'Breakpoint Value':breakpoints[1:], 'Initial Count':initial_counts, 'New Count':new_counts})
df['Initial Percent'] = df['Initial Count'] / len(initial)
df['New Percent'] = df['New Count'] / len(new)
df['New Percent'][df['New Percent'] == 0] = 0.001
print(df)
# percents = df[['Initial Percent', 'New Percent', 'Bucket']] \
#              .melt(id_vars=['Bucket']) \
#              .rename(columns={'variable':'Population', 'value':'Percent'})
#
# p = sns.barplot(x="Bucket", y="Percent", hue="Population", data=percents)
# p.set(xlabel='Bucket', ylabel='Population Percent')
# sns.despine(left=True)
# plt.show()

df['PSI'] = (df['New Percent'] - df['Initial Percent']) * np.log(df['New Percent'] / df['Initial Percent'])
print(df)

print(np.round(calculate_psi(initial, new, buckets=10, axis=1), 5))
print(np.round(np.sum(df['PSI']), 5))

breakpoints = np.stack([np.percentile(initial, b) for b in np.arange(0, buckets + 1) / (buckets) * 100])
initial_counts = np.histogram(initial, breakpoints)[0]
new_counts = np.histogram(new, breakpoints)[0]
print(initial_counts)
print(new_counts)
# df = pd.DataFrame({'Bucket': np.arange(1, 11), 'Breakpoint Value':breakpoints[1:], 'Initial Count':initial_counts, 'New Count':new_counts})
# df['Initial Percent'] = df['Initial Count'] / len(initial)
# df['New Percent'] = df['New Count'] / len(new)
# df['New Percent'][df['New Percent'] == 0] = 0.001
# percents = df[['Initial Percent', 'New Percent', 'Bucket']] \
#              .melt(id_vars=['Bucket']) \
#              .rename(columns={'variable':'Population', 'value':'Percent'})
#
# p = sns.barplot(x="Bucket", y="Percent", hue="Population", data=percents)
# p.set(xlabel='Bucket', ylabel='Population Percent')
# sns.despine(left=True)


print(calculate_psi(initial, new, buckettype='quantiles', buckets=10, axis=1))








