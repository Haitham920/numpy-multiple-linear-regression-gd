# Multiple Linear Regression from Scratch (NumPy + Gradient Descent)

A multiple linear regression trainer written in *pure NumPy* — no scikit-learn, no autograd. It standardizes features, minimizes mean squared error with *full-batch gradient descent and early stopping*, checks the result against the closed-form *normal equation*, and reports *MAE, RMSE and R²* through a small fit / predict / score API.

The goal is to understand every piece of the pipeline by writing it by hand, one function at a time.

## Results

On the synthetic demo data in scaffold.py (150 samples, 3 features, Gaussian noise σ = 0.1):

| Metric (test set) | Value |
|---|---|
| MAE | 0.092 |
| RMSE | 0.122 |
| R² | 0.9985 |
| L2 gap between GD and normal-equation weights | 0.0053 |
| Epochs before early stopping | 81 / 400 |

Gradient descent lands within 0.005 (L2 distance) of the exact least-squares solution, and early stopping halts training once the validation loss stops improving.

## Quick start

Requires Python 3 and NumPy.

git clone https://github.com/Haitham920/numpy-multiple-linear-regression-gd.git
cd numpy-multiple-linear-regression-gd
pip install numpy
python scaffold.py

Expected output:

Splits: 90 30 30
Sample preds: [-4.9361 -2.1618 -1.1091 -3.5921  1.7049]
Sample trues: [-4.8688 -2.2391 -0.8235 -3.5582  1.6763]
Test MAE/RMSE/R2: {'mae': 0.0921..., 'rmse': 0.1219..., 'r2': 0.9985...}
GD vs normal-eq L2 gap: 0.00526...

## Using the model on your own data

import numpy as np
from model import (shuffle_xy, split_train_val_test, create_lr_model,
                   fit_lr_model, predict_lr_model, score_lr_model,
                   compare_with_normal_equation)

X, y = shuffle_xy(X, y, seed=42)
X_tr, y_tr, X_val, y_val, X_te, y_te = split_train_val_test(X, y, 0.6, 0.2)

model = create_lr_model(learning_rate=0.05, epochs=400, patience=25, seed=0)
model = fit_lr_model(model, X_tr, y_tr, X_val, y_val)

y_hat   = predict_lr_model(model, X_te)        # takes raw (unscaled) features
metrics = score_lr_model(model, X_te, y_te)    # {'mae', 'rmse', 'r2'}
gap     = compare_with_normal_equation(model)  # ||w_gd - w_closed||_2

The model is a plain dictionary holding the hyperparameters, the learned weights, the normal-equation weights, the training-set mean/std used for scaling, and the per-epoch train/validation losses.

## How it works

### 1. Data preparation
- *Shuffle* rows of X and y with one shared seeded permutation.
- *Split* into contiguous train / validation / test partitions (60 / 20 / 20 by default).
- *Standardize* each feature using the training mean and standard deviation only, so no information leaks from the validation or test sets. Constant features (std = 0) get std = 1 to avoid division by zero.
- *Add a bias column* of ones, so the intercept is learned as the first weight.

### 2. Model and loss

$$\hat{y} = Xw \qquad \text{MSE}(w) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

### 3. Gradient descent

$$\nabla_w \text{MSE} = \frac{2}{n} X^\top(\hat{y} - y) \qquad w \leftarrow w - \eta \, \nabla_w \text{MSE}$$

Weights start from $\mathcal{N}(0,\,0.01)$. Each epoch is one full-batch update, followed by computing train and validation MSE.

### 4. Early stopping
The weights with the lowest validation loss so far are kept. If validation loss fails to improve for patience consecutive epochs, training stops and the best weights are returned rather than the last ones.

### 5. Closed-form check

$$w^{*} = X^{+} y$$

computed with the Moore–Penrose pseudo-inverse (np.linalg.pinv), which stays stable even when $X^\top X$ is singular. The L2 distance between the GD weights and $w^{*}$ shows how close the iterative solution got to the exact one.

### 6. Evaluation
- *MAE* — mean absolute error
- *RMSE* — root mean squared error, in the same units as the target
- *R²* — share of the target's variance explained by the model (returns NaN if the target is constant)

## Project structure

.
├── model.py        # All 28 functions: data prep, training, metrics, model API
├── scaffold.py     # End-to-end demo on synthetic data
└── docs/
    └── index.html  # Project write-up page

## Function reference

| Stage | Functions |
|---|---|
| Data preparation | shuffle_xy, split_train_val_test, compute_feature_stats, standardize_features, add_bias_column, prepare_design_matrix |
| Core math | predict_linear, mse_loss, mse_gradient, normal_equation |
| Training loop | initialize_weights, gd_step, epoch_train_val_losses, update_early_stop_state, init_training_state, run_one_epoch, train_batch_gd |
| Metrics | mean_absolute_error, root_mean_squared_error, r_squared, evaluate_regression, learning_curve_data, weights_l2_distance |
| Model API | create_lr_model, fit_lr_model, predict_lr_model, score_lr_model, compare_with_normal_equation |

## Possible extensions

- Plot learning curves with matplotlib using learning_curve_data
- Mini-batch or stochastic gradient descent
- L2 (ridge) or L1 (lasso) regularization
- Benchmark against sklearn.linear_model.LinearRegression on a real dataset (e.g. California Housing)

## Acknowledgements

Built step by step following the [Deep-ML](https://www.deep-ml.com) project track.

## Author

*Haitham Maatar* — [GitHub](https://github.com/Haitham920)
github.com
