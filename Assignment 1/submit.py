import numpy as np
import sklearn

# You are allowed to import any submodules of numpy or sklearn e.g. sklearn.metrics.accuracy_score to calculate accuracy of a learnt model
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_map, my_params etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here

################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

    # Transforming {0, 1} to {-1, 1}
    # Here, X_mapped will have the same shape as X, but with values in {-1, 1} instead of {0, 1}
    X_transformed = 1-2*X

    X_even = X_transformed[:, 0::2]  # Nx16 shaped vector consisting of the even indexed features (0, 2, 4, ..., 30)
    X_odd = X_transformed[:, 1::2]   # Nx16 shaped vector consisting of the odd indexed features (1, 3, 5, ..., 31)

    # TRANSFORMED VECTOR: [ x1*x2  x1*x3 ... x1*x32  x2*x3 x2*x4 ... x31*x32 | x0 x2 ... x32 | x1 x3 ... x31 ]
    # Structure of the vector: [ PAIRWISE PRODUCTS | EVEN | ODD ]

    # Computing the pairwise products of the even and odd indexed features
    pairwise_products = (X_even[:, :, None] * X_odd[:, None, :]).reshape(X.shape[0], -1)  # Final shape: Nx256

    # Stacking the pairwise products with the original features to get the final mapped vector
    X_map = np.hstack((pairwise_products, X_even, X_odd))  # Final shape: Nx288 (288 is the dimension "D" from part2)

    return X_map

################################
# Non Editable Region Starting #
################################
def my_params( X_map, X_raw, y ):
################################
#  Non Editable Region Ending  #
################################

	my_params = {
        "loss"     : "squared_hinge",
        "C"        : 0.5,
        "tol"      : 0.01,
        "dual"     : "auto",
        "max_iter" : 5000,
	}
	
	return my_params

