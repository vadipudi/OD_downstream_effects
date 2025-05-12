import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

def logistic_regression_classifier(train_features, test_features, train_target):
    # Create a Logistic Regression Classifier
    classifier = LogisticRegression()

    # Train the classifier on the training data
    classifier.fit(train_features, train_target)

    # Predict the classes for the testing data
    test_predictions = classifier.predict(test_features)

    return test_predictions
def decision_tree_classifier(train_features, test_features, train_target):


    # Create a Decision Tree Classifier
    classifier = DecisionTreeClassifier()

    # Train the classifier on the training data
    classifier.fit(train_features, train_target)

    # Predict the classes for the testing data
    test_predicionts = classifier.predict(test_features)

    return test_predicionts



def random_forest_classifier(train_features, test_features, train_target):
    # Create a Random Forest Classifier
    classifier = RandomForestClassifier()

    # Train the classifier on the training data
    classifier.fit(train_features, train_target)

    # Predict the classes for the testing data
    test_predictions = classifier.predict(test_features)

    return test_predictions


def support_vector_classifier(train_features, test_features, train_target):


    # Create a Support Vector Classifier
    classifier = SVC()

    # Train the classifier on the training data
    classifier.fit(train_features, train_target)

    # Predict the classes for the testing data
    test_predictions = classifier.predict(test_features)

    return test_predictions



def knn_classifier(train_features, test_features, train_target):
    # Create a K-Nearest Neighbors Classifier
    classifier = KNeighborsClassifier()

    # Train the classifier on the training data
    classifier.fit(train_features, train_target)

    # Predict the classes for the testing data
    test_predictions = classifier.predict(test_features)

    return test_predictions


def gradient_boosting_classifier(train_features, test_features, train_target):
    # Create a Gradient Boosting Classifier
    classifier = GradientBoostingClassifier()

    # Train the classifier on the training data
    classifier.fit(train_features, train_target)

    # Predict the classes for the testing data
    test_predictions = classifier.predict(test_features)

    return test_predictions