"""
NumPy Multiple Linear Regression GD

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - shuffle_xy
import numpy as np
def shuffle_xy(X, y, seed=42):
    """Randomly permute feature rows and targets together.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    seed : int, optional
        RNG seed for reproducibility (default 42).

    Returns
    -------
    X_shuffled : np.ndarray, shape (n, d)
    y_shuffled : np.ndarray, shape (n,)
    """
    # TODO: Return (X, y) under one shared seeded row permutation
    n=X.shape[0]
    idx = np.random.default_rng(seed).permutation(n)
    X_shuffled=X[idx]
    y_shuffled=y[idx]
    return(X_shuffled,y_shuffled)
    pass

# Step 2 - split_train_val_test
import numpy as np
def split_train_val_test(X, y, train_frac=0.6, val_frac=0.2):
    # TODO: Slice already-shuffled data into contiguous train/val/test partitions...
    n=X.shape[0]
    n_train=int(n * train_frac)
    n_val=n_train + int(n * val_frac)
    X_train=X[:n_train]
    y_train=y[:n_train]
    X_val=X[n_train:n_val]
    y_val=y[n_train:n_val]
    X_test=X[n_val:]
    y_test=y[n_val:]
    return(X_train,y_train,X_val,y_val,X_test,y_test)
    pass

# Step 3 - compute_feature_stats
import numpy as np 
def compute_feature_stats(X):
    # TODO: Compute per-feature mean and std; replace std of 0 with 1
    mean_x=np.mean(X,axis=0)
    std_X=np.std(X,axis=0)
    mask=std_X == 0
    std_X[mask]=1
    return (mean_x,std_X)
    pass

# Step 4 - standardize_features
import numpy as np
def standardize_features(X, mean, std):
    # TODO: Apply z-score normalization using precomputed training mean and std.
    n=X.shape[1]
    X=X.T
    for i in range (n):
        X[i]=(X[i]-mean[i])/std[i]
    X=X.T
    return X
    pass

# Step 5 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to feature matrix X
    n,m=X.shape
    X=np.insert(X, 0, np.ones(len(X)), axis=1)
    return X

# Step 6 - prepare_design_matrix
import numpy as np
def prepare_design_matrix(X, mean, std):
    # TODO: Standardize features then add the bias column to form the design matrix.
    X=standardize_features(X, mean, std)
    X=add_bias_column(X)
    return X
    pass

# Step 7 - predict_linear
import numpy as np
def predict_linear(X, weights):
    """Compute linear predictions y_hat = X @ weights.

    Args:
        X: Design matrix of shape (n, d_in), often including a bias column.
        weights: Weight vector of shape (d_in,).

    Returns:
        Predicted targets of shape (n,).
    """
    # TODO: Return the predicted target vector from X and weights
    if (X.shape[1]==weights.shape[0]):
        return X @ weights
    raise ValueError("the shapes does not match")
    pass

# Step 8 - mse_loss
import numpy as np
def mse_loss(y_true, y_pred):
    # TODO: Return the average of squared residuals as a scalar float.
    n=y_true.shape[0]
    return (1/n)*np.sum((y_true-y_pred)**2)
    pass

# Step 9 - mse_gradient
import numpy as np
def mse_gradient(X, y_true, y_pred):
    # TODO: Return the analytic MSE gradient w.r.t. weights: (2/n) X^T (y_pred - y_true)
    n=X.shape[0]
    return (2/n)* X.T @ (y_pred-y_true)
    pass

# Step 10 - normal_equation
import numpy as np
def normal_equation(X, y):
    # TODO: Solve for the closed-form least-squares weights via the normal equation.
    A=X.T @ X
    b=X.T @ y
    return np.linalg.solve(A,b)
    pass

# Step 11 - initialize_weights
import numpy as np
def initialize_weights(n_features, seed=None):
    # TODO: Return (n_features,) weights sampled from N(0, 0.01)
    if (seed != None):
        np.random.seed(seed)
    return(np.random.normal(0,0.01,n_features))

# Step 12 - gd_step
def gd_step(X, y, weights, lr):
    """Run one full-batch gradient descent update on the weights.

    Args:
        X: Design matrix of shape (n, d_in).
        y: Target vector of shape (n,).
        weights: Current weight vector of shape (d_in,).
        lr: Learning rate (float).

    Returns:
        Updated weight vector of shape (d_in,).
    """
    # TODO: return the updated weight vector after one MSE gradient step
    y_pred=predict_linear(X,weights)
    mse=mse_gradient(X,y, y_pred)
    return(weights-lr*mse)
    pass

# Step 13 - epoch_train_val_losses
def epoch_train_val_losses(X_train, y_train, X_val, y_val, weights):
    """Evaluate MSE on train and validation sets for the current weights.

    Args:
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        weights: Weight vector of shape (d_in,).

    Returns:
        (train_loss, val_loss) as plain floats.
    """
    # TODO: return the pair (train_loss, val_loss) as MSE floats
    y_pred_train=predict_linear(X_train,weights)
    y_pred_val=predict_linear(X_val,weights)
    train_loss=mse_loss(y_train,y_pred_train)
    val_loss=mse_loss(y_val,y_pred_val)
    return(train_loss,val_loss)

    
    pass

# Step 14 - update_early_stop_state
def update_early_stop_state(val_loss, best_val_loss, wait, weights, best_weights, patience):
    # TODO: Update best weights and patience counter; signal stop when val loss stalls...
    if(val_loss<best_val_loss):
        best_val_loss=val_loss
        best_weights= weights.copy()
        wait=0
    else:
        wait+=1
    flag=(wait == patience)
    return(best_val_loss,wait,best_weights,flag)
    pass

# Step 15 - init_training_state
import numpy as np
def init_training_state(n_features, seed=None):
    # TODO: Build the initial training-state dictionary for the GD epoch loop.
    weights=initialize_weights(n_features,seed)
    return{'weights':weights,'best_weights':weights.copy(),'best_val_loss':np.inf ,'wait': 0,'train_losses':[],'val_losses':[],'stopped':False}
    pass

# Step 16 - run_one_epoch
def run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience):
    """Perform one GD step, log losses, and refresh early-stopping on state.

    Args:
        state: Dict with keys weights, best_weights, best_val_loss, wait,
            stopped, train_losses, val_losses.
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        lr: Learning rate (float).
        patience: Early-stopping patience (int).

    Returns:
        Updated state dict.
    """
    # TODO: Take one GD step, log train/val losses, refresh early-stopping fields...
    new_w=gd_step(X_train,y_train,state['weights'],lr)
    A=epoch_train_val_losses(X_train,y_train,X_val,y_val,new_w)
    B=update_early_stop_state(A[1],state['best_val_loss'],state['wait'],new_w,state['best_weights'],patience)
    state['weights']=new_w
    state['best_weights']=B[2]
    state['best_val_loss']=B[0]
    state['wait']=B[1]
    state['stopped']=B[3]
    state['train_losses'].append (A[0])
    state['val_losses'].append(A[1])
    return state

# Step 17 - train_batch_gd
import numpy as np 
def train_batch_gd(X_train, y_train, X_val, y_val, lr, epochs, patience, seed=None):
    state=init_training_state(X_train.shape[1],seed)
    i=0
    while((state['stopped']!= True)and(i<epochs)):
        state=run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience)
        i+=1
    return(state['best_weights'],state['train_losses'],state['val_losses'])
    pass

# Step 18 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true-y_pred))
    pass

# Step 19 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    # TODO: Return the root mean squared error between y_true and y_pred.
    return float(np.sqrt((1/len(y_true))*np.sum((y_true-y_pred)**2)))
    pass

# Step 20 - r_squared
import numpy as np
def r_squared(y_true, y_pred):
    # TODO: Compute the coefficient of determination R^2.
    SST=0
    SSR=0
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mean= np.mean(y_true)
    for i in range (len(y_true)):
        SSR+=((y_true[i]-y_pred[i])**2)
        SST+=((y_true[i]-mean)**2)  
    if SST == 0.0:
        return np.nan
        
    return 1.0 - (SSR / SST)
    pass

# Step 21 - evaluate_regression
import numpy as np
def evaluate_regression(y_true, y_pred):
    # TODO: Bundle MAE, RMSE, and R^2 into one metrics dictionary for test-set reporting.
    return {'mae': mean_absolute_error(y_true, y_pred),'rmse': root_mean_squared_error(y_true, y_pred),'r2': r_squared(y_true, y_pred)}
    pass

# Step 22 - learning_curve_data
import numpy as np

def learning_curve_data(train_losses, val_losses):
    n = len(train_losses)
    train_list = np.asarray(train_losses).tolist()
    val_list = np.asarray(val_losses).tolist()
    
    # Return train_list and val_list instead of train_losses and val_losses
    return (list(range(1, n + 1)), (train_list), (val_list))

# Step 23 - weights_l2_distance
def weights_l2_distance(w_gd, w_closed):
    # TODO: Compute the L2 distance between two weight vectors
    return np.linalg.norm(w_gd-w_closed)
    pass

# Step 24 - create_lr_model
def create_lr_model(learning_rate=0.01, epochs=1000, patience=50, seed=0):
    # TODO: Build the initial LinearRegressionGD-style model dictionary...
    return {'learning_rate':learning_rate,'epochs':epochs,'patience':patience,'seed':seed,'weights':None,'normal_weights':None,'mean':None,'std':None,'train_losses':[],'val_losses':[]}
    pass

# Step 25 - fit_lr_model
import numpy as np

def compute_feature_stats(X):
    """
    Compute feature-wise mean and standard deviation on training data.
    Replace zero std values with 1.0 to avoid division by zero.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0.0] = 1.0
    return mean, std

def prepare_design_matrix(X, mean, std):
    """
    Standardize features using given mean and std, then PREPEND a bias column of ones.
    Output shape: (n_samples, d + 1)
    """
    X_scaled = (X - mean) / std
    ones = np.ones((len(X), 1), dtype=float)
    return np.hstack([ones, X_scaled])  # Prepend column of ones (bias)

def normal_equation(X, y):
    """
    Compute closed-form weights using pseudo-inverse (np.linalg.pinv) 
    to handle singular/collinear design matrices robustly.
    """
    return np.linalg.pinv(X) @ y

def train_batch_gd(X_train, y_train, X_val, y_val, lr, epochs, patience, seed=None):
    """
    Train Linear Regression using Batch Gradient Descent with early stopping.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
        weights = rng.normal(0, 0.01, size=X_train.shape[1])
    else:
        weights = np.zeros(X_train.shape[1], dtype=float)
        
    train_losses = []
    val_losses = []
    
    best_weights = weights.copy()
    best_val_loss = float('inf')
    patience_counter = 0
    
    n_train = len(X_train)
    
    for epoch in range(epochs):
        # Forward pass & loss
        y_train_pred = X_train @ weights
        train_loss = float(np.mean((y_train - y_train_pred) ** 2))
        train_losses.append(train_loss)
        
        y_val_pred = X_val @ weights
        val_loss = float(np.mean((y_val - y_val_pred) ** 2))
        val_losses.append(val_loss)
        
        # Gradient computation & update
        gradient = (-2 / n_train) * (X_train.T @ (y_train - y_train_pred))
        weights -= lr * gradient
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = weights.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    return best_weights, train_losses, val_losses

def fit_lr_model(model, X_train, y_train, X_val, y_val):
    """
    Fit a linear regression model dict using training data only.
    Updates and returns the model dictionary.
    """
    # 1. Compute feature statistics from X_train ONLY
    mean, std = compute_feature_stats(X_train)          

    # 2. Build design matrices with prepended bias column
    X_train_design = prepare_design_matrix(X_train, mean, std)
    X_val_design = prepare_design_matrix(X_val, mean, std)   

    # 3. Train via Batch Gradient Descent using stored hyperparameters
    best_weights, train_losses, val_losses = train_batch_gd(
        X_train_design, y_train, X_val_design, y_val,
        model['learning_rate'], model['epochs'], model['patience'], seed=model['seed'],
    )
    
    # 4. Compute closed-form normal equation weights
    normal_weights = normal_equation(X_train_design, y_train)

    # 5. Populate model dictionary
    model['mean'] = mean
    model['std'] = std
    model['weights'] = best_weights
    model['normal_weights'] = normal_weights
    model['train_losses'] = train_losses
    model['val_losses'] = val_losses
    
    return model

# Step 26 - predict_lr_model
def predict_lr_model(model, X):
    # TODO: Return predicted targets for raw X using the fitted model.
    X_stand=prepare_design_matrix(X,model['mean'],model['std'])
    return predict_linear(X_stand,model['weights'])
    pass

# Step 27 - score_lr_model
def score_lr_model(model, X, y):
    # TODO: Predict on raw features and return MAE, RMSE, and R^2 metrics.
    y_predict=predict_lr_model(model, X)
    return evaluate_regression(y,y_predict)
    pass

# Step 28 - compare_with_normal_equation
def compare_with_normal_equation(model):
    # TODO: Return the L2 distance between GD and normal-equation weights.
    return weights_l2_distance(model['weights'],model['normal_weights'])
    pass

