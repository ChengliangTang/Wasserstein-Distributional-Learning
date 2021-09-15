# This scripts run simulations in a bootstrap way
# import packages
import os, sys
sys.path.append('../../../lib/')
import numpy as np
import pandas as pd
import WDL as wp
from datetime import datetime

# set parameters
SEED = int(os.getenv('SLURM_ARRAY_TASK_ID'))
K = 2
n_dist = 200 ## number of distributions
omega = 0.1 # default 0.1
n_sample = 300

# simulate the data 
np.random.seed(SEED)
X = np.random.uniform(size=(n_dist, 3)) * 2 - 1
Y = np.zeros((n_dist, n_sample))

## simulate Y
for i in range(n_dist):
    mu_1 = X[i, 0]
    mu_2 = 2 * X[i, 1]**2 + 2
    mu_true = [mu_1, mu_2]
    sig_1 = np.abs(X[i, 1]) + 0.5
    sig_2 = np.abs(X[i, 0]) + 0.5
    sig_true = [sig_1, sig_2]
    pi_1 = 1 / (1 + np.exp(X[i, 2]))
    pi_true = [pi_1, 1-pi_1]
    ## simulate noise
    eps_noise = np.random.normal(loc=0, scale=omega, size=1)
    ## simulate responses
    var_gaussian = np.array([np.random.normal(loc=mu_true[k]+eps_noise, 
                                              scale=sig_true[k], 
                                              size=n_sample) for k in range(K)]).T
    var_mult = np.random.choice(range(K), size=n_sample, replace=True, p=pi_true)
    var_mult = np.eye(K)[var_mult]
    var_GMM = np.sum(var_mult * var_gaussian, axis=1)
    Y[i] = np.sort(var_GMM)

# prepare for model fitting
K_mix = 2
n_iter = 1000
lr = 1e-1
v_lr = np.array([1] + [lr] * n_iter)
n_dist = Y.shape[0]
n_levs = 100
q_vec = np.arange(1, n_levs) / n_levs
## transform Y
Q_mat = np.array([np.quantile(Y[i], q_vec) for i in range(n_dist)])

# step 1. train-val split
n_train = int(0.8 * n_dist) 
loc_train = np.random.choice(n_dist, n_train, replace=False)
loc_val = np.setdiff1d(np.arange(n_dist), loc_train)

# step 2. model training with early stop
time_start = datetime.now()
print('Start training:', time_start)
res_init = wp.WDL(X[loc_train], Q_mat[loc_train], X[loc_val], Q_mat[loc_val], q_vec=q_vec, K=K_mix, max_iter=n_iter, warm_up=1, max_depth=1, patience=5, lr=lr, random_state=SEED)
print('Time:', datetime.now() - time_start)   

# step 3. model training with the best number of steps
best_iter = res_init['iter_best']
res = wp.WDL(X, Q_mat, X, Q_mat, q_vec=q_vec, K=K_mix, max_iter=best_iter, warm_up=1, max_depth=1, early_stop=False, lr=lr, random_state=SEED)




# create partial dependence plot
n_evals = 21
x_eval = np.linspace(-1, 1, n_evals)
X_eval = np.array(np.meshgrid(x_eval, x_eval, x_eval)).T.reshape(-1,3)
p_levs = [0.1, 0.3, 0.5, 0.7, 0.9]
par_mat = np.zeros((3, n_evals, 5))
res_mat =  np.zeros((3, n_evals, 5))
print('Start PDP calculation:', datetime.now())
for i in range(3):
    for j in range(n_evals):
        X_c = X_eval.copy()
        X_c[:, i] = x_eval[j]
        alpha_pred = np.zeros((X_eval.shape[0], K_mix))
        mu_pred = np.zeros((X_eval.shape[0], K_mix))
        sigma_pred = np.zeros((X_eval.shape[0], K_mix))
        for k in range(K_mix):
            alpha_pred[:, k] = wp.pred_boost(X_c, res['alpha'][k], lr_=v_lr, n_term=res['iter_best'])
            mu_pred[:, k] = wp.pred_boost(X_c, res['mu'][k], lr_=v_lr, n_term=res['iter_best'])
            sigma_pred[:, k] = np.exp(wp.pred_boost(X_c, res['sigma'][k], lr_=v_lr, n_term=res['iter_best']))
        pi_pred = np.exp(alpha_pred)
        pi_pred = (pi_pred.T / np.sum(pi_pred, axis=1)).T
        par_mat[i, j, 0] = np.mean(pi_pred[:, 0])
        par_mat[i, j, 1:3] = np.mean(mu_pred, axis=0)
        par_mat[i, j, 3:5] = np.mean(sigma_pred, axis=0)
        res_mat[i, j] = np.mean([wp.qgmm1d(p_levs, mu_pred[l], sigma_pred[l], pi_pred[l]) for l in range(n_dist)], axis=0)
## save the results
np.save('output/bootstrap/par_' + str(SEED).zfill(3) + '.npy', par_mat)
np.save('output/bootstrap/qt_' + str(SEED).zfill(3) + '.npy', res_mat)
print('Done! Total time:', datetime.now() - time_start) 
