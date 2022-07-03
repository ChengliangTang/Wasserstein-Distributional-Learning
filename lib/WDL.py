import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn import *

# density of 1D Gaussian Mixture
def dgmm1d(x, mu, sigma, pi):
    pdf_gmm = np.sum([pi[k] * norm.pdf(x, loc=mu[k], scale=sigma[k]) for k in range(len(mu))], axis=0)
    return pdf_gmm

# cdf of 1D Gaussian Mixture
def pgmm1d(x, mu, sigma, pi):
    cdf_gmm = np.sum([pi[k] * norm.cdf(x, loc=mu[k], scale=sigma[k]) for k in range(len(mu))], axis=0)
    return cdf_gmm

# quantile of 1D Gaussian Mixture
def qgmm1d(q, mu, sigma, pi):
    ppf_full = np.array([norm.ppf(q, loc=mu[k], scale=sigma[k]) for k in range(len(mu))]).flatten()
    ppf_full.sort()
    cdf_gmm = np.sum([pi[k] * norm.cdf(ppf_full, loc=mu[k], scale=sigma[k]) for k in range(len(mu))], axis=0)
    ## 1D linear interpolation
    ppf_gmm = np.interp(q, cdf_gmm, ppf_full)
    return ppf_gmm

# 1D Wasserstein distance between raw data and GMM
def dWasserstein(x, mu, sigma, pi, q, p=2):
    '''
    :param array x: vector of raw data (sorted)
    :param array mu: vector of component means
    :param array sigma: vector of component standard deviations (positive)
    :param array pi: vector of component weights (\in [0, 1])
    :param array q: vector of quantiles (corresponding to x)
    :param integer p: order of Wasserstein distance
    '''
    ppf_gmm = qgmm1d(q, mu, sigma, pi)
    Wp = np.mean((ppf_gmm - x)**p) ** (1/p)
    return Wp

# make boosted predictions
def pred_boost(input_x, model_list, lr_, n_term):
    '''
    :param array input_x: array of input features
    :param list model_list: list of different models
    :param array lr: array of learning rate in each step (same length with model_list)
    '''
    preds = np.array([model.predict(input_x) for model in model_list])
    output_y = np.sum([preds[:n_term, i] * lr_[:n_term] for i in range(input_x.shape[0])], axis=1)
    return output_y

# main functions for fitting Wasserstein mixture regression
def WDL(X_train, Y_train, X_val, Y_val, q_vec, K=2, init='EM', warm_up=30, 
        max_iter=300, max_depth=1, lr=1e-1, loss_crit='absolute_error', early_stop=True, 
        patience=5, random_state=0):
    """  
    :param array X_train: array of scaler training input
    :param array Y_train: array of distributional training output
    :param array X_val: array of scaler validation input
    :param array Y_val: array of distributional validation output
    :param 1-D array q_vec: array of corresponding quantile levels
    :param int K: number of mixtures
    :param int max_iter: number of iterations
    :param float lr: learning rate
    :param boolean early_stop: whether to use early stopping
    :param patience: patience in early stopping
    """
    np.random.seed(random_state) ## set random state
    eps = 1e-10
    n_train = Y_train.shape[0]
    n_val = Y_val.shape[0]
    n_lev = len(q_vec)
    tol = 0 ## early stopping
    if not early_stop:
        patience = max_iter
    
    ## array for storing predictions
    mu_train = np.zeros((n_train, K))
    mu_val = np.zeros((n_val, K))
    log_sd_train = np.zeros((n_train, K))
    log_sd_val = np.zeros((n_val, K))
    alpha_train = np.zeros((n_train, K))
    alpha_val = np.zeros((n_val, K))
    
    ## list for storing models
    models_mu_ = [[] for k in range(K)]
    models_sd_ = [[] for k in range(K)]
    models_alpha_ = [[] for k in range(K)]
    
    ## initialization
    grad_mu = np.sort(np.random.rand(n_train, K)) - 0.5
    grad_sd = np.random.rand(n_train, K) - 0.5
    grad_alpha = np.random.rand(n_train, K) - 0.5
    grad_alpha[:, 0] = 0
    
    loss_train_ = []
    loss_val_ = []
    
    if init == 'EM':
        for i in range(n_train):
            gmm = mixture.GaussianMixture(n_components=K)
            gmm.fit(np.reshape(Y_train[i], (-1, 1)))
            id_sort = np.argsort(gmm.means_.flatten())
            grad_mu[i] = gmm.means_.flatten()[id_sort]
            grad_sd[i] = np.sqrt(gmm.covariances_.flatten())[id_sort]
            grad_sd[i] = np.log(grad_sd[i])
            grad_alpha[i] = np.log(gmm.weights_.flatten())[id_sort]
            grad_alpha[i] = grad_alpha[i] - grad_alpha[i][0]
            
    ## warm-up: GBM for EM fitting
    for k in range(K):
        grad_mu_s = grad_mu[:, k]
        grad_sd_s = grad_sd[:, k]
        grad_alpha_s = grad_alpha[:, k]
        ## create the model
        gbm_mu = ensemble.GradientBoostingRegressor(learning_rate=lr, n_estimators=warm_up, random_state=random_state)
        gbm_sd = ensemble.GradientBoostingRegressor(learning_rate=lr, n_estimators=warm_up, random_state=random_state)
        gbm_alpha = ensemble.GradientBoostingRegressor(learning_rate=lr, n_estimators=warm_up, random_state=random_state)
        ## fit the model
        gbm_mu.fit(X_train, grad_mu_s)
        gbm_sd.fit(X_train, grad_sd_s)
        gbm_alpha.fit(X_train, grad_alpha_s)
        ## add the model to model list
        models_mu_[k].append(gbm_mu)
        models_sd_[k].append(gbm_sd)
        models_alpha_[k].append(gbm_alpha)
        ## make predictions on training set
        mu_train[:, k] = gbm_mu.predict(X_train)
        log_sd_train[:, k] = gbm_sd.predict(X_train)
        alpha_train[:, k] = gbm_alpha.predict(X_train)
        ## make predictions on test set
        mu_val[:, k] = gbm_mu.predict(X_val)
        log_sd_val[:, k] = gbm_sd.predict(X_val)
        alpha_val[:, k] = gbm_alpha.predict(X_val)
    
    sd_train = np.exp(log_sd_train)
    sd_val = np.exp(log_sd_val)
    
    ## generate \pi array
    pi_train = np.exp(alpha_train)
    pi_train = (pi_train.T / np.sum(pi_train, axis=1)).T
    pi_val = np.exp(alpha_val)
    pi_val = (pi_val.T / np.sum(pi_val, axis=1)).T
    ## updates are done! save the training log
    w2_train = np.mean([dWasserstein(Y_train[j], mu_train[j], sd_train[j], pi_train[j], q_vec)**2 for j in range(n_train)])
    w2_val = np.mean([dWasserstein(Y_val[j], mu_val[j], sd_val[j], pi_val[j], q_vec)**2 for j in range(n_val)])
    
    loss_train_.append(w2_train)
    loss_val_.append(w2_val)
    
    ## start training
    early_exit = False
    for i in range(max_iter):
        ## step 1. update \alpha
        for j in range(n_train):
            R = np.array([alpha_train[j, k] + norm.logpdf(Y_train[j], mu_train[j, k], sd_train[j, k]) for k in range(K)])
            R = R - np.max(R, axis=0)
            R = np.exp(R)
            R += eps ## prevent exploding
            R = R / np.sum(R, axis=0)
            ## update \pi
            N_ks = np.sum(R, axis = 1)
            pi_Opt = N_ks / n_lev
            alpha_Opt = np.log(pi_Opt) - np.log(pi_Opt)[0]
            grad_alpha[j] = alpha_Opt - alpha_train[j]
        ## minimize fitted loss with criterion
        for k in range(K):
            grad_alpha_s = grad_alpha[:, k]
            ## create model
            clf_alpha = tree.DecisionTreeRegressor(criterion=loss_crit, max_depth=max_depth)
            ## fit models
            clf_alpha.fit(X_train, grad_alpha_s)
            ## add to the list
            models_alpha_[k].append(clf_alpha)
            ## update \alpha array
            alpha_train[:, k] += lr * clf_alpha.predict(X_train)
            alpha_val[:, k] += lr * clf_alpha.predict(X_val)
        ## generate \pi array
        pi_train = np.exp(alpha_train)
        pi_train = (pi_train.T / np.sum(pi_train, axis=1)).T
        pi_val = np.exp(alpha_val)
        pi_val = (pi_val.T / np.sum(pi_val, axis=1)).T
        ## step 2 + 3. find the optimal decomposition of g, and then update \mu and \sigma 
        for j in range(n_train):
            y_hat = qgmm1d(q_vec, mu_train[j], sd_train[j], pi_train[j])
            R = np.array([alpha_train[j, k] + norm.logpdf(Y_train[j], mu_train[j, k], sd_train[j, k]) for k in range(K)])
            R = R - np.max(R, axis=0)
            R = np.exp(R)
            R += eps ## prevent exploding
            R = (R / np.sum(R, axis=0)).T
            ## find the optimal \mu and \sigma given the current \alpha and \g
            Y_array = np.tile(Y_train[j], (K, 1)).T
            #P = (np.cumsum(R, axis=0) / np.sum(R, axis=0) - 0.5) * (q_vec[-1] - q_vec[0]) + 0.5
            P = np.cumsum(R, axis=0) / np.sum(R, axis=0) * n_lev/(n_lev+1)
            Z = norm.ppf(P)
            A = np.sum(R, axis=0)
            B = np.sum(R * Z, axis=0)
            C = np.sum(R * Y_array, axis=0)
            E = B
            F = np.sum(R * Z**2, axis=0)
            D = np.sum(R * Y_array * Z, axis=0)
            mu_Opt = (C*F - B*D) / (A*F - B*E + eps)
            sigma_Opt = np.abs((A*D - C*E)) / (A*F - B*E + eps) + eps ## prevent exploding
            # avoid identifiability by sorting the \mu
            id_sort = np.argsort(mu_Opt)
            pi_train[j] = pi_train[j][id_sort]
            alpha_train[j] = alpha_train[j][id_sort]
            grad_mu[j] = mu_Opt[id_sort] - mu_train[j]
            grad_sd[j] = np.log(sigma_Opt[id_sort]) - log_sd_train[j]
        ## fit boosted stumps to update \mu and \sigma
        for k in range(K):
            grad_mu_s = grad_mu[:, k]
            grad_sd_s = grad_sd[:, k]
            ## create model
            clf_mu = tree.DecisionTreeRegressor(criterion=loss_crit, max_depth=max_depth)
            clf_sigma = tree.DecisionTreeRegressor(criterion=loss_crit, max_depth=max_depth)
            ## fit models
            clf_mu.fit(X_train, grad_mu_s)
            clf_sigma.fit(X_train, grad_sd_s)
            ## add to the list
            models_mu_[k].append(clf_mu)
            models_sd_[k].append(clf_sigma)
            ## update the array
            mu_train[:, k] += lr * clf_mu.predict(X_train)
            log_sd_train[:, k] += lr * clf_sigma.predict(X_train)
            ## make predictions on validation set
            mu_val[:, k] += lr * clf_mu.predict(X_val)
            log_sd_val[:, k] += lr * clf_sigma.predict(X_val)
        sd_train = np.exp(log_sd_train)
        sd_val = np.exp(log_sd_val)
        ## updates are done! save the training log
        w2_train = np.mean([dWasserstein(Y_train[j], mu_train[j], sd_train[j], pi_train[j], q_vec)**2 for j in range(n_train)])
        w2_val = np.mean([dWasserstein(Y_val[j], mu_val[j], sd_val[j], pi_val[j], q_vec)**2 for j in range(n_val)])
        
        if w2_val < loss_val_[-1]:
            tol = 0
        elif early_stop:
            tol += 1
            
        if tol < patience:
            loss_train_.append(w2_train)
            loss_val_.append(w2_val)
        else:
            early_exit = True
            break
            
    if early_exit:
        iter_best = np.argmin(np.array(loss_val_))
    else:
        iter_best = max_iter
    
    ## return outputs
    outputs = {'iter_best': iter_best, 'alpha': models_alpha_, 
               'mu': models_mu_, 'sigma': models_sd_, 
               'train_loss': loss_train_, 'val_loss': loss_val_}
    return outputs
        
    
            
# main functions for fitting EM Regression
def EMR(X_train, Y_train, X_val, Y_val, q_vec, K=2, init='EM', warm_up=30, 
        max_iter=300, max_depth=1, lr=1e-1, loss_crit='absolute_error', early_stop=True, 
        patience=5, random_state=0):
    """  
    :param array X_train: array of scaler training input
    :param array Y_train: array of distributional training output
    :param array X_val: array of scaler validation input
    :param array Y_val: array of distributional validation output
    :param 1-D array q_vec: array of corresponding quantile levels
    :param int K: number of mixtures
    :param int max_iter: number of iterations
    :param float lr: learning rate
    :param boolean early_stop: whether to use early stopping
    :param patience: patience in early stopping
    """
    np.random.seed(random_state) ## set random state
    eps = 1e-10
    n_train = Y_train.shape[0]
    n_val = Y_val.shape[0]
    n_sample = len(q_vec)
    tol = 0 ## early stopping
    if not early_stop:
        patience = max_iter
        
    ## prediction initializations
    mu_train = np.zeros((n_train, K))
    mu_val = np.zeros((n_val, K))
    log_sd_train = np.zeros((n_train, K))
    log_sd_val = np.zeros((n_val, K))
    alpha_train = np.zeros((n_train, K))
    alpha_val = np.zeros((n_val, K))
    
    ## list for storing models
    models_mu_ = [[] for k in range(K)]
    models_sd_ = [[] for k in range(K)]
    models_alpha_ = [[] for k in range(K)]

    ## initialization
    grad_mu = np.sort(np.random.rand(n_train, K)) - 0.5
    grad_sd = np.random.rand(n_train, K) - 0.5
    grad_alpha = np.random.rand(n_train, K) - 0.5
    grad_alpha[:, 0] = 0
    
    loss_train_LL = []
    loss_val_LL = []
    loss_train_WL = []
    loss_val_WL = []
    
    if init == 'EM':
        for i in range(n_train):
            gmm = mixture.GaussianMixture(n_components=K)
            gmm.fit(np.reshape(Y_train[i], (-1, 1)))
            id_sort = np.argsort(gmm.means_.flatten())
            grad_mu[i] = gmm.means_.flatten()[id_sort]
            grad_sd[i] = np.sqrt(gmm.covariances_.flatten())[id_sort]
            grad_sd[i] = np.log(grad_sd[i])
            grad_alpha[i] = np.log(gmm.weights_.flatten())[id_sort]
            grad_alpha[i] = grad_alpha[i] - grad_alpha[i][0]
            
    ## warm-up: GBM for EM fitting
    for k in range(K):
        grad_mu_s = grad_mu[:, k]
        grad_sd_s = grad_sd[:, k]
        grad_alpha_s = grad_alpha[:, k]
        ## create the model
        gbm_mu = ensemble.GradientBoostingRegressor(learning_rate=lr, n_estimators=warm_up, random_state=random_state)
        gbm_sd = ensemble.GradientBoostingRegressor(learning_rate=lr, n_estimators=warm_up, random_state=random_state)
        gbm_alpha = ensemble.GradientBoostingRegressor(learning_rate=lr, n_estimators=warm_up, random_state=random_state)
        ## fit the model
        gbm_mu.fit(X_train, grad_mu_s)
        gbm_sd.fit(X_train, grad_sd_s)
        gbm_alpha.fit(X_train, grad_alpha_s)
        ## add the model to model list
        models_mu_[k].append(gbm_mu)
        models_sd_[k].append(gbm_sd)
        models_alpha_[k].append(gbm_alpha)
        ## make predictions on training set
        mu_train[:, k] = gbm_mu.predict(X_train)
        log_sd_train[:, k] = gbm_sd.predict(X_train)
        alpha_train[:, k] = gbm_alpha.predict(X_train)
        ## make predictions on test set
        mu_val[:, k] = gbm_mu.predict(X_val)
        log_sd_val[:, k] = gbm_sd.predict(X_val)
        alpha_val[:, k] = gbm_alpha.predict(X_val)
        
    sd_train = np.exp(log_sd_train)
    sd_val = np.exp(log_sd_val)
    
    ## generate \pi array
    pi_train = np.exp(alpha_train)
    pi_train = (pi_train.T / np.sum(pi_train, axis=1)).T
    pi_val = np.exp(alpha_val)
    pi_val = (pi_val.T / np.sum(pi_val, axis=1)).T
    ## updates are done! save the training log
    w2_train = np.mean([dWasserstein(Y_train[j], mu_train[j], sd_train[j], pi_train[j], q_vec)**2 for j in range(n_train)])
    w2_val = np.mean([dWasserstein(Y_val[j], mu_val[j], sd_val[j], pi_val[j], q_vec)**2 for j in range(n_val)])
    ll_train = -np.log([dgmm1d(Y_train[j], mu_train[j], sd_train[j], pi_train[j])**2 + eps for j in range(n_train)])
    ll_val = -np.log([dgmm1d(Y_val[j], mu_val[j], sd_val[j], pi_val[j])**2 + eps for j in range(n_val)])
    
    loss_train_WL.append(w2_train)
    loss_val_WL.append(w2_val)
    loss_train_LL.append(np.mean(ll_train))
    loss_val_LL.append(np.mean(ll_val))
    
    ## start training
    early_exit = False
    for i in range(max_iter):
        ## step 1. update \alpha
        for j in range(n_train):
            R = np.array([alpha_train[j, k] + norm.logpdf(Y_train[j], mu_train[j, k], sd_train[j, k]) for k in range(K)])
            R = R - np.max(R, axis=0)
            R = np.exp(R)
            R += eps ## prevent exploding
            R = R / np.sum(R, axis=0)
            ## update \pi
            N_ks = np.sum(R, axis = 1)
            pi_Opt = N_ks / n_sample 
            alpha_Opt = np.log(pi_Opt) - np.log(pi_Opt)[0]
            grad_alpha[j] = alpha_Opt - alpha_train[j]
        ## minimize fitted loss with criterion
        for k in range(K):
            grad_alpha_s = grad_alpha[:, k]
            ## create model
            clf_alpha = tree.DecisionTreeRegressor(criterion=loss_crit, max_depth=1)
            ## fit models
            clf_alpha.fit(X_train, grad_alpha_s)
            ## add to the list
            models_alpha_[k].append(clf_alpha)
            ## update \alpha array
            alpha_train[:, k] += lr * clf_alpha.predict(X_train)
            alpha_val[:, k] += lr * clf_alpha.predict(X_val)
        ## generate \pi array
        pi_train = np.exp(alpha_train)
        pi_train = (pi_train.T / np.sum(pi_train, axis=1)).T
        pi_val = np.exp(alpha_val)
        pi_val = (pi_val.T / np.sum(pi_val, axis=1)).T
        ## step 2 + 3. find the optimal decomposition of g, and then update \mu and \sigma 
        for j in range(n_train):
            R = np.array([alpha_train[j, k] + norm.logpdf(Y_train[j], mu_train[j, k], sd_train[j, k]) for k in range(K)])
            R = R - np.max(R, axis=0)
            R = np.exp(R)
            R += eps ## prevent exploding
            R = (R / np.sum(R, axis=0)).T
            ## find the optimal \mu and \sigma given the current \alpha and \g
            Y_array = np.tile(Y_train[j], (K, 1)).T
            P = (np.cumsum(R, axis=0) / np.sum(R, axis=0) - 0.5) * (q_vec[-1] - q_vec[0]) + 0.5
            Z = norm.ppf(P)
            A = np.sum(R, axis=0)
            B = np.sum(R * Z, axis=0)
            C = np.sum(R * Y_array, axis=0)
            E = B
            F = np.sum(R * Z**2, axis=0)
            D = np.sum(R * Y_array * Z, axis=0)
            mu_Opt = C / A
            sigma_Opt = np.sqrt(np.sum(R * (Y_array - mu_Opt)**2, axis=0) / A) + eps ## prevent explodin
            # avoid identifiability by sorting the \mu
            id_sort = np.argsort(mu_Opt)
            pi_train[j] = pi_train[j][id_sort]
            alpha_train[j] = alpha_train[j][id_sort]
            grad_mu[j] = mu_Opt[id_sort] - mu_train[j]
            grad_sd[j] = np.log(sigma_Opt[id_sort]) - log_sd_train[j]
        ## fit boosted stumps to update \mu and \sigma
        for k in range(K):
            grad_mu_s = grad_mu[:, k]
            grad_sd_s = grad_sd[:, k]
            ## create model
            clf_mu = tree.DecisionTreeRegressor(criterion=loss_crit, max_depth=1)
            clf_sigma = tree.DecisionTreeRegressor(criterion=loss_crit, max_depth=1)
            ## fit models
            clf_mu.fit(X_train, grad_mu_s)
            clf_sigma.fit(X_train, grad_sd_s)
            ## add to the list
            models_mu_[k].append(clf_mu)
            models_sd_[k].append(clf_sigma)
            ## update the array
            mu_train[:, k] += lr * clf_mu.predict(X_train)
            log_sd_train[:, k] += lr * clf_sigma.predict(X_train)
            ## make predictions on validation set
            mu_val[:, k] += lr * clf_mu.predict(X_val)
            log_sd_val[:, k] += lr * clf_sigma.predict(X_val)
        sd_train = np.exp(log_sd_train)
        sd_val = np.exp(log_sd_val)
        ## updates are done! save the training log
        w2_train = np.mean([dWasserstein(Y_train[j], mu_train[j], sd_train[j], pi_train[j], q_vec)**2 for j in range(n_train)])
        w2_val = np.mean([dWasserstein(Y_val[j], mu_val[j], sd_val[j], pi_val[j], q_vec)**2 for j in range(n_val)])
        ll_train = -np.log([dgmm1d(Y_train[j], mu_train[j], sd_train[j], pi_train[j])**2 + eps for j in range(n_train)])
        ll_val = -np.log([dgmm1d(Y_val[j], mu_val[j], sd_val[j], pi_val[j])**2 + eps for j in range(n_val)])
        
        if np.mean(ll_val) < loss_val_LL[-1]:
            tol = 0
        elif early_stop:
            tol += 1
            
        if tol < patience:
            loss_train_WL.append(w2_train)
            loss_val_WL.append(w2_val)
            loss_train_LL.append(np.mean(ll_train))
            loss_val_LL.append(np.mean(ll_val))
        else:
            early_exit = True
            break
            
    if early_exit:
        iter_best = np.argmin(np.array(loss_val_LL))
    else:
        iter_best = max_iter
  
    ## return outputs
    outputs = {'iter_best': iter_best, 'alpha': models_alpha_, 
               'mu': models_mu_, 'sigma': models_sd_, 
               'train_loss_WL': loss_train_WL, 'val_loss_WL': loss_val_WL,
               'train_loss_LL': loss_train_LL, 'val_loss_LL': loss_val_LL}
    return outputs
            
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    






