import numpy as np

# Parameters
n_samples = 1000  # Number of samples
n_features = 20   # Number of features
d = n_features    # Dimensionality

# Generate synthetic feature vectors from a Gaussian distribution
X = np.random.normal(0, 1, (n_samples, n_features))

# Define a non-linear decision function (e.g., a radial function)
def decision_function(x):
    center = np.ones(n_features)  # Example center for the radial function
    radius = np.linalg.norm(x - center)
    return radius

# Assign labels based on the decision function
labels = np.array([1 if decision_function(x) > np.linalg.norm(center) else 0 for x in X])

# Your X and labels are now ready to be used in training
