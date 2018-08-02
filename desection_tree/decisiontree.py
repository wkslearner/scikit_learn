from desection_tree.id3_c45 import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, auc,classification_report
from sklearn.tree import DecisionTreeClassifier


if __name__ == '__main__':
    # Toy data
    X = [[1, 2, 0, 1, 0],
         [0, 1, 1, 0, 1],
         [1, 0, 0, 0, 1],
         [2, 1, 1, 0, 1],
         [1, 1, 0, 1, 1]]
    y = ['yes', 'yes', 'no', 'no', 'no']

    clf = DecisionTree(mode='ID3')
    clf.fit(X, y)
    #clf.show()
    #precision, recall, thresholds = precision_recall_curve(y, y_predict)
    #classification_report(y, y_predict>0.5, target_names=['neg', 'pos'])
    print (clf.predict(X))  #['yes' 'yes' 'no' 'no' 'no']

    clf_ = DecisionTree(mode='C4.5')
    clf_.fit(X, y)
    #clf_.show()
    print(clf_.predict(X))  #['yes' 'yes' 'no' 'no' 'no']
