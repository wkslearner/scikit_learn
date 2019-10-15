

import jieba
from gensim import corpora,models,similarities

all_doc=['北京市朝阳区建国门外大街','北京市建国门外大街5号院',
         '浙江省金华市武义县浙江省金华市武义县武义县武义县温泉路77号',
         '北京市朝阳区外大街5号院',
         '天津市天津市武清区天津市武清区天津市武清区逸仙科技园祥云超市祥云超市']


# 1 分词
# 1.1 历史比较文档的分词
all_doc_list = []
for doc in all_doc:
    doc_list = [word for word in jieba.cut_for_search(doc)]
    print(doc_list)











