import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

def multiple_linear_regression(train_features, test_features, train_target):
    """
    Perform multiple linear regression.

    Multiple linear regression is a regression model that examines the linear relationship between multiple independent variables and a dependent variable.
    It assumes that the relationship between the independent variables and the dependent variable is linear.

    Args:
        train_features (DataFrame): Training features.
        test_features (DataFrame): Testing features.
        train_target (Series): Training target.
        test_target (Series): Testing target.

    Returns:
        tuple: A tuple containing the predicted target values and the actual target values.
    """
    # Train the linear regression model
    model = LinearRegression()
    model.fit(train_features, train_target)

    # Predict the target values for the testing data
    test_predictions = model.predict(test_features)

    return test_predictions


def ridge_regression(train_features, test_features, train_target, alpha=1.0):
    """
    Perform ridge regression.

    Ridge regression is a regression model that adds a penalty term to the ordinary least squares method to reduce the impact of multicollinearity in the data.
    It uses L2 regularization to shrink the coefficients towards zero, reducing the model's complexity.

    Args:
        train_features (DataFrame): Training features.
        test_features (DataFrame): Testing features.
        train_target (Series): Training target.
        test_target (Series): Testing target.
        alpha (float): Regularization strength.

    Returns:
        tuple: A tuple containing the predicted target values and the actual target values.
    """
    # Train the ridge regression model
    model = Ridge(alpha=alpha)
    model.fit(train_features, train_target)

    # Predict the target values for the testing data
    test_predictions = model.predict(test_features)

    return test_predictions


def lasso_regression(train_features, test_features, train_target, alpha=1.0):
    """
    Perform lasso regression.

    Lasso regression is a regression model that adds a penalty term to the ordinary least squares method to reduce the impact of multicollinearity in the data.
    It uses L1 regularization to shrink some coefficients to exactly zero, effectively performing feature selection.

    Args:
        train_features (DataFrame): Training features.
        test_features (DataFrame): Testing features.
        train_target (Series): Training target.
        test_target (Series): Testing target.
        alpha (float): Regularization strength.

    Returns:
        tuple: A tuple containing the predicted target values and the actual target values.
    """
    # Train the lasso regression model
    model = Lasso(alpha=alpha)
    model.fit(train_features, train_target)

    # Predict the target values for the testing data
    test_predictions = model.predict(test_features)

    return test_predictions


def elastic_net_regression(train_features, test_features, train_target, alpha=1.0, l1_ratio=0.5):
    """
    Perform elastic net regression.

    Elastic net regression is a regression model that combines the properties of ridge regression and lasso regression.
    It adds both L1 and L2 regularization terms to the ordinary least squares method, allowing for feature selection and reducing the impact of multicollinearity.

    Args:
        data_file (str): Path to the CSV file containing the data.
        test_size (float): The proportion of the data to be used for testing.
        alpha (float): Regularization strength.
        l1_ratio (float): The mixing parameter between L1 and L2 regularization.

    Returns:
        tuple: A tuple containing the predicted target values and the actual target values.
    """

    # Train the elastic net regression model
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(train_features, train_target)

    # Predict the target values for the testing data
    test_predictions = model.predict(test_features)

    return test_predictions


def polynomial_regression(train_features, test_features, train_target,  degree=2):
    """
    Perform polynomial regression.

    Polynomial regression is a regression model that extends the linear regression model by adding polynomial terms to the features.
    It allows for modeling non-linear relationships between the independent variables and the dependent variable.

    Args:
        data_file (str): Path to the CSV file containing the data.
        test_size (float): The proportion of the data to be used for testing.
        degree (int): The degree of the polynomial features.

    Returns:
        tuple: A tuple containing the predicted target values and the actual target values.
    """

    # Create polynomial features
    polynomial_features = PolynomialFeatures(degree=degree)
    train_features_poly = polynomial_features.fit_transform(train_features)
    test_features_poly = polynomial_features.transform(test_features)

    # Train the polynomial regression model
    model = LinearRegression()
    model.fit(train_features_poly, train_target)

    # Predict the target values for the testing data
    test_predictions = model.predict(test_features_poly)

    return test_predictions


def random_forest_regression(train_features, test_features, train_target, n_estimators=100):
    """
    Perform random forest regression.

    Random forest regression is an ensemble regression model that combines multiple decision trees to make predictions.
    It reduces overfitting and improves prediction accuracy by averaging the predictions of multiple trees.

    Args:
        data_file (str): Path to the CSV file containing the data.
        test_size (float): The proportion of the data to be used for testing.
        n_estimators (int): The number of trees in the random forest.

    Returns:
        tuple: A tuple containing the predicted target values and the actual target values.
    """

    # Train the random forest regression model
    model = RandomForestRegressor(n_estimators=n_estimators)
    model.fit(train_features, train_target)

    # Predict the target values for the testing data
    test_predictions = model.predict(test_features)

    return test_predictions


def support_vector_regression(train_features, test_features, train_target, kernel='rbf', C=1.0, epsilon=0.1):
    """
    Perform support vector regression.

    Support vector regression is a regression model that uses support vector machines to perform regression.
    It finds a hyperplane that maximizes the margin between the predicted values and the actual values, while allowing for a certain amount of error.

    Args:
        data_file (str): Path to the CSV file containing the data.
        test_size (float): The proportion of the data to be used for testing.
        kernel (str): Specifies the kernel type to be used in the algorithm.
        C (float): Regularization parameter.
        epsilon (float): Epsilon in the epsilon-SVR model.

    Returns:
        tuple: A tuple containing the predicted target values and the actual target values.
    """

    # Train the support vector regression model
    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    model.fit(train_features, train_target)

    # Predict the target values for the testing data
    test_predictions = model.predict(test_features)

    return test_predictions


# Example usage



# data_file = '/Users/vikramadipudi/Desktop/Thesis_research/Workspace/test.csv'
# predictions,test_target = multiple_linear_regression(train_features, test_features, train_target, test_size=0.2)
# mse = mean_squared_error(test_target, predictions)
# print('Mean Squared Error:', mse)