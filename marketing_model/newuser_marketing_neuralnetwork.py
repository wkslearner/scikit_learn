#!/usr/bin/python
# encoding=utf-8

import pandas as pd
import tensorflow as tf
from data_process.list_process import remove_list
from data_process.feature_handle import disper_split
from sklearn.model_selection import train_test_split
import argparse
from sklearn import preprocessing


tf.logging.set_verbosity(tf.logging.INFO)

'''设定参数'''
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=5000, type=int,
                    help='number of training steps')



'''构建dataset对象的训练数据'''
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Build the Iterator, and return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


newuser_dataset=pd.read_excel('/Users/andpay/Documents/job/model/newuser_marketing_credithands/newuser_marketing_dataset_v4_1.xlsx')
var_list=list(newuser_dataset.columns)
model_var_list=remove_list(var_list,['partyid','cate'])

category_var=['sex','city-id','channel','brandcode']
continue_var=remove_list(model_var_list,category_var)
newuser_dataset=disper_split(newuser_dataset,category_var)
newuser_dataset[continue_var]=newuser_dataset[continue_var].fillna(-1)

newuser_dataset=newuser_dataset[model_var_list+['cate']].apply(pd.to_numeric)

#自变量标准化处理，可以减少训练时间
newuser_dataset[model_var_list]=preprocessing.scale(newuser_dataset[model_var_list])


#固定训练集和测试集数据
traindata, testdata= train_test_split(newuser_dataset,test_size=0.25,random_state=1)
x_train,y_train=traindata[model_var_list],traindata['cate']
x_test,y_test=testdata[model_var_list],testdata['cate']


#创建tesorflow可用的特征形式
my_feature_columns = []
for key in x_train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))


classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Three hidden layers of 10 nodes.
    hidden_units=[10, 10, 10,10],
    # The model must choose between 2 classes.
    n_classes=2)


args = parser.parse_args()

#进行模型训练
classifier.train(
    input_fn=lambda: train_input_fn(x_train, y_train, args.batch_size),
    steps=args.train_steps)

print('train complete')

# Evaluate the model.
eval_result = classifier.evaluate(
    input_fn=lambda: eval_input_fn(x_train, y_train, args.batch_size))
print('\nTest set accuracy: {accuracy:0.4f}\n'.format(**eval_result))

predictions = classifier.predict(
    input_fn=lambda: eval_input_fn(x_test, labels=None, batch_size=args.batch_size))

for i in predictions:
    print(i)


prediction_ids = [prediction['class_ids'][0] for prediction in predictions]


submission = pd.DataFrame({
    "PassengerId": y_test,
    "Survived": prediction_ids
})












