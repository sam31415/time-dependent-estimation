import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from pyfinance.ols import OLS, RollingOLS, PandasRollingOLS
import pykalman
from pykalman import KalmanFilter
from scipy.interpolate import splrep, splev
from scipy.stats import wilcoxon
import multiprocessing as mp
from functools import partial

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


# Preparation
# -----------

# Sensitivities
def get_sensitivities(timespan, sens_type):
    pi = 3.1415
    if sens_type == "Various smooth":
        t = np.linspace(0,timespan-1,timespan,dtype=np.float64)
        sens1 = -(pd.Series(np.arctan((t-timespan/2)/timespan*100)) - pi/2)/pi
        sens2 = (pd.Series(np.sin(t/timespan*3.2 - 1.5)) + 1)/2
        sens3 = pd.Series(np.sin(t/timespan*5) + np.sin(t/timespan*10 + 150) + \
                          np.sin(t/timespan*20 + 150))/2
    elif sens_type == "Random walk":
        sens1 = pd.Series(np.random.normal(size=timespan)).cumsum()/200 + 1
        sens2 = pd.Series(np.random.normal(size=timespan)).cumsum()/200
        sens3 = pd.Series(np.random.normal(size=timespan)).cumsum()/200 - 1
    elif sens_type == "Random smooth":
        sens1 = create_random_sensitivity(timespan)
        sens2 = create_random_sensitivity(timespan)
        sens3 = create_random_sensitivity(timespan)

    return pd.concat([sens1, sens2, sens3], axis=1)

    
# Random sensitivities
def create_random_sensitivity(timespan):
    num_points = int(np.round(np.random.rand()*8) + 2)
    smoothing_parameter = np.exp(2*np.random.rand() - 1.5)
    x = np.floor(np.random.rand(num_points)*timespan)
    x = np.append(x, [0, timespan-1])
    x.sort()
    y = 2*np.random.rand(num_points + 2) - 1

    
    spleen = splrep(x, y, s=smoothing_parameter)
    t = np.linspace(0,timespan-1,timespan,dtype=np.float64)
    values = splev(t, spleen)
    sensitivity = pd.Series(data=values, index=t)
    sensitivity[sensitivity < -5] = -5
    sensitivity[sensitivity > 5] = 5
    return sensitivity
    

# Factors
def get_factors(timespan):
    factor1 = pd.Series(np.random.normal(size=timespan))
    factor2 = pd.Series(np.random.normal(size=timespan))
    factor3 = pd.Series(np.random.normal(size=timespan))
    return pd.concat([factor1, factor2, factor3], axis=1)
    
    
# Returns
def get_returns(timespan, factors, sens, error_std):
    error = np.random.normal(size=timespan)*error_std
    return pd.Series(np.multiply(factors, sens).sum(axis=1) + error)


# Model estimation
# ----------------

def estimate_sensitivities_cr(factors, returns):
    reg = linear_model.LinearRegression()
    reg.fit(factors, returns)
    return pd.DataFrame(data = [reg.coef_], index = factors.index)
    
    
def estimate_sensitivities_rr(factors, returns, window):
    model = PandasRollingOLS(y=returns, x=factors, window=window)
    return pd.DataFrame(model.beta)
    
    
def estimate_sensitivities_err(factors, returns, lambd):
    ff = np.zeros([3,3])
    fr = np.zeros(3)
    est_sens_list = []
    for i, row in factors.iterrows():
        ff = (1-lambd)*ff + lambd*np.outer(row, row)
        fr = (1-lambd)*fr + lambd*np.dot(row, returns.iloc[i])
        est_sens = np.dot(np.linalg.inv(ff + 1e-9*np.eye(3)), fr)
        est_sens_list.append(est_sens)
    return pd.DataFrame(est_sens_list)  
    
    
def estimate_sensitivities_kf(factors, returns, covariance_ratio):
    n_dim_state=3
    n_dim_obs=1
    observation_covariance = 1
    transition_covariance = np.eye(3)*covariance_ratio
    observation_matrices = factors.values[:,np.newaxis,:]
    kf = KalmanFilter(initial_state_mean=[0,0,0], 
                      n_dim_state=n_dim_state, 
                      n_dim_obs=n_dim_obs, 
                      observation_matrices=observation_matrices,
                      observation_covariance=observation_covariance,
                      transition_covariance=transition_covariance
                     )
    filtered_state_means, filtered_state_covariances = kf.filter(returns)
    return pd.DataFrame(filtered_state_means)
    

def estimate_sensitivities_ntc(estimated_sensitivities, trend_factor, window):
    rolling_mean = estimated_sensitivities.rolling(window, axis=0).mean()
    return (1 + trend_factor)*pd.DataFrame(estimated_sensitivities) - trend_factor*rolling_mean
    
    
def estimate_sensitivities_stkf(factors, returns, covariance_ratio):
    n_dim_state=6
    n_dim_obs=1
    observation_covariance = 1
    transition_covariance = covariance_ratio*np.eye(6)
    observation_matrices = np.array([factors.values[:, 0], 0*factors.values[:, 0],
                                     factors.values[:, 1], 0*factors.values[:, 1],
                                     factors.values[:, 2], 0*factors.values[:, 2]]).transpose()[:, np.newaxis,:]
    transition_matrix = np.kron(np.eye(3), np.array([[1, 1], [0, 1]]))
    kf = KalmanFilter(initial_state_mean=[0, 0, 0, 0, 0, 0], 
                      n_dim_state=n_dim_state, 
                      n_dim_obs=n_dim_obs, 
                      observation_matrices=observation_matrices,
                      observation_covariance=observation_covariance,
                      transition_covariance=transition_covariance,
                      transition_matrices=transition_matrix
                     )
    filtered_state_means, filtered_state_covariances = kf.filter(returns)
    return pd.DataFrame(filtered_state_means[:,[0,2,4]])
    

# Evaluation
# ----------
    
def plot_estimated_sensitivities(sens, est_sens, colors):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    axc = ax[0,0]
    axc.plot(est_sens.iloc[:,0], c=colors[0])
    axc.plot(sens.iloc[:,0], '--', c='k')
    axc = ax[0,1]
    axc.plot(est_sens.iloc[:,1], c=colors[1])
    axc.plot(sens.iloc[:,1], '--', c='k')
    axc = ax[1,0]
    axc.plot(est_sens.iloc[:,2], c=colors[2])
    axc.plot(sens.iloc[:,2], '--', c='k')
  
  
def compute_estimated_returns(factors, est_sens):
    return pd.Series(np.multiply(factors, est_sens.shift()).sum(axis=1))

    
def plot_residuals(estimated_returns, returns):
    residuals = estimated_returns - returns
    plt.scatter(residuals.index, residuals, s=0.5)
   
   
def print_mse_performance(estimated_returns, returns, name, performance_record_mse, verbose=True):
    square_errors = np.square(estimated_returns - returns)
    mean_square_error = square_errors.sum()/len(estimated_returns)
    performance_record_mse[name] = mean_square_error
    if verbose:
        print(f"MSE for {name}, assuming the factors known: {mean_square_error}")
    
    
def print_weighted_accuracy_performance(estimated_returns, returns, name, performance_record_wacc, verbose=True):
    weights = abs(returns)
    correct_sign = -2*np.logical_xor(estimated_returns > 0, returns > 0) + 1
    weighted_accuracy = np.multiply(weights, correct_sign).mean()
    performance_record_wacc[name] = weighted_accuracy
    if verbose:
        print(f"Weighted accuracy for {name}, assuming the factors known: {weighted_accuracy}")
   
   
def plot_prediction_performance(factors, returns, est_sens, name, 
        performance_record_mse, performance_record_wacc):
    estimated_returns = compute_estimated_returns(factors, est_sens)
    print_mse_performance(estimated_returns, returns, name, performance_record_mse)
    print_weighted_accuracy_performance(estimated_returns, returns, name, performance_record_wacc)
    plot_residuals(estimated_returns, returns)
  

def estimate_regression_performance(factors, returns, name, performance_record_mse, 
                                    performance_record_wacc, **kwargs):
    burn_period = 500
    if name[:12] == "Constant OLS":
        start_time = burn_period
        estimated_sensitivities = estimate_sensitivities_cr(factors, returns)
        estimated_sensitivities = estimated_sensitivities.iloc[start_time:,:]
    elif name[:11] == "Rolling OLS":
        start_time = burn_period
        rolling_ols_window = kwargs['rolling_ols_window']
        estimated_sensitivities = estimate_sensitivities_rr(factors, returns, rolling_ols_window)
        estimated_sensitivities = estimated_sensitivities.iloc[burn_period - rolling_ols_window + 1:,:]
    elif name[:23] == "Exponential rolling OLS":
        start_time = burn_period
        exp_rolling_ols_lambda = kwargs['exp_rolling_ols_lambda']
        estimated_sensitivities = estimate_sensitivities_err(factors, returns, 
                                                             exp_rolling_ols_lambda)
        estimated_sensitivities = estimated_sensitivities.iloc[start_time:,:]
    elif name[:18] == "Kalman naive trend":
        start_time = burn_period
        kalman_covariance_ratio = kwargs['kalman_covariance_ratio']
        nt_factor = kwargs['nt_factor']
        nt_window = kwargs['nt_window']
        estimated_sensitivities_kf = estimate_sensitivities_kf(factors, returns, 
                                                               kalman_covariance_ratio)
        estimated_sensitivities = estimate_sensitivities_ntc(estimated_sensitivities_kf, 
                                                             nt_factor, nt_window)
        estimated_sensitivities = estimated_sensitivities.iloc[start_time:,:]
    elif name[:18] == "Kalman stoch trend":
        start_time = burn_period
        stkf_covariance_ratio = kwargs['stkf_covariance_ratio']
        estimated_sensitivities = estimate_sensitivities_stkf(factors, returns, stkf_covariance_ratio)
        estimated_sensitivities = estimated_sensitivities.iloc[start_time:,:]
    elif name[:6] == "Kalman":
        start_time = burn_period
        kalman_covariance_ratio = kwargs['kalman_covariance_ratio']
        estimated_sensitivities = estimate_sensitivities_kf(factors, returns, kalman_covariance_ratio)
        estimated_sensitivities = estimated_sensitivities.iloc[start_time:,:]
    else:
        print("Unknown name")
    estimated_returns = compute_estimated_returns(factors.iloc[start_time:,:], estimated_sensitivities)
    print_mse_performance(estimated_returns, 
                          returns.iloc[start_time:], 
                          name, 
                          performance_record_mse,
                          verbose=False)
    print_weighted_accuracy_performance(estimated_returns, 
                                        returns.iloc[start_time:], 
                                        name, 
                                        performance_record_wacc,
                                        verbose=False) 

def compare_hyperparameters_mp(algo_name, num_samples, timespan, sens_type, **kwargs):
    """
    The keyword arguments passed should correspond to the parameters of the algorithms and consists of lists of values to 
    try (a grid search).
    """
    parameter_list = []
    value_list = []
    list_performance_record_mse = []
    list_performance_record_wacc = []
    n_processes = 7
    
    for parameter, par_value_list in kwargs.items():
        parameter_list.append(parameter)
        value_list.append(par_value_list)
    all_values = list(itertools.product(*value_list))
    
    for i in range(len(all_values)):
        all_values[i] = dict(zip(parameter_list, all_values[i]))
    
    func_parallel = partial(generate_sample_hyperparameters, sens_type, timespan, algo_name, all_values)
    
    with mp.Pool(n_processes) as pool:
        outputs = pool.imap_unordered(func_parallel, range(num_samples))
        for out in outputs:
            print(out[0])
            list_performance_record_mse.append(out[1])
            list_performance_record_wacc.append(out[2])
    
    return pd.DataFrame(list_performance_record_mse), pd.DataFrame(list_performance_record_wacc)

def generate_sample_hyperparameters(sens_type, timespan, algo_name, all_values, i):
    print(i)
    performance_record_mse = {}
    performance_record_wacc = {}
    # Sensitivities
    if sens_type in ["Random walk", "Random smooth"]:
        sens = get_sensitivities(timespan, sens_type)
    # Factors
    factors = get_factors(timespan)
    # Returns
    returns = get_returns(timespan, factors, sens, 1.0)

    for kwargs in all_values:
        name = algo_name + ' - ' + str(kwargs)
        estimate_regression_performance(factors, returns, name, performance_record_mse, performance_record_wacc, **kwargs)
    return i, performance_record_mse, performance_record_wacc
   
   
def get_performance_data(num_samples, sens_type, timespan, algo_parameters, sens=None):
    list_performance_record_mse = []
    list_performance_record_wacc = []
    n_processes = 5
    
    func_parallel = partial(generate_sample_performance, sens_type, timespan, algo_parameters, sens)
    
    with mp.Pool(n_processes) as pool:
        outputs = pool.imap_unordered(func_parallel, range(num_samples))
        for out in outputs:
            print(out[0])
            list_performance_record_mse.append(out[1])
            list_performance_record_wacc.append(out[2])
    
    return pd.DataFrame(list_performance_record_mse), pd.DataFrame(list_performance_record_wacc)
    

def generate_sample_performance(sens_type, timespan, algo_parameters, sens, i):
    performance_record_mse = {}
    performance_record_wacc = {}
    rolling_ols_window = algo_parameters['rolling_ols_window']
    exp_rolling_ols_lambda = algo_parameters['exp_rolling_ols_lambda']
    kalman_covariance_ratio = algo_parameters['kalman_covariance_ratio']
    stkf_covariance_ratio = algo_parameters['stkf_covariance_ratio']
           
    # Sensitivities
    if sens_type in ["Random walk", "Random smooth"]:
        sens = get_sensitivities(timespan, sens_type)
    # Factors
    factors = get_factors(timespan)
    # Returns
    returns = get_returns(timespan, factors, sens, 1.0)
            
    estimate_regression_performance(factors, returns, "Constant OLS", 
                                    performance_record_mse, performance_record_wacc)
    estimate_regression_performance(factors, returns, "Rolling OLS", 
                                    performance_record_mse, performance_record_wacc, rolling_ols_window=rolling_ols_window)
    estimate_regression_performance(factors, returns, "Exponential rolling OLS", 
                                    performance_record_mse, performance_record_wacc, 
                                    exp_rolling_ols_lambda=exp_rolling_ols_lambda)
    estimate_regression_performance(factors, returns, "Kalman", 
                                    performance_record_mse, performance_record_wacc, 
                                    kalman_covariance_ratio=kalman_covariance_ratio)
    #estimate_regression_performance(factors, returns, "Kalman naive trend", 
    #                                performance_record_mse, performance_record_wacc, 
    #                                kalman_covariance_ratio=kalman_covariance_ratio, 
    #                                nt_factor=nt_factor, nt_window=nt_window)
    estimate_regression_performance(factors, returns, "Kalman stoch trend", 
                                    performance_record_mse, performance_record_wacc, 
                                    stkf_covariance_ratio=stkf_covariance_ratio)
    return i, performance_record_mse, performance_record_wacc
    
    
def print_t_stat(performance, reference=None):
    if reference is None:
        ref_column = performance.iloc[:,0]
    else:
        ref_column = performance.loc[:,reference]
    rel_performance = performance.subtract(ref_column, axis = 0)
    t_stat = rel_performance.mean().divide(rel_performance.std()/np.sqrt(len(rel_performance)))
    if reference is None:
        t_stat.iloc[0] = 0
    else:
        t_stat.loc[reference] = 0
    return t_stat
    
def plot_t_stat_differences(performance, reference=None):
    if reference is None:
        ref_column = performance.iloc[:,0]
        purged_performance = performance.iloc[:,1:]
    else:
        ref_column = performance.loc[:,reference]
        purged_performance = performance.drop([reference], axis=1)
    rel_performance = purged_performance.subtract(ref_column, axis = 0)
    rel_performance.hist(layout=(purged_performance.shape[1], 1), bins=50, sharex=True, sharey=True)
    plt.tight_layout()