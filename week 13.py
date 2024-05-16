# Classification Example: Logistic Regression for Iris Flower Classification
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate the model
print(f"Accuracy: {clf.score(X_test, y_test)}")

# Regression Example: Linear Regression for Boston Housing Prices
from sklearn import datasets
from sklearn.linear_model import LinearRegression

# Load the Boston housing dataset
boston = datasets.load_boston()
X, y = boston.data, boston.target

# Create and train the linear regression model
reg = LinearRegression()
reg.fit(X, y)

# Evaluate the model
print(f"R-squared: {reg.score(X, y)}")

# Clustering Example: K-Means Clustering
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate sample data
X, _ = make_blobs(n_samples=1000, centers=5, n_features=2, random_state=42)

# Create and fit the K-Means clustering model
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)

# Get the cluster labels
labels = kmeans.labels_