# model_dispatcher.py

from sklearn import tree
from sklearn import ensemble

models = {'decision_tree_gini': tree.DecisionTreeClassifier(criterion="gini"),
          'decision_tree_entropy': tree.DecisionTreeClassifier(criterion=\
                                       "entropy"),
          'random_forest': ensemble.RandomForestClassifier()
}
