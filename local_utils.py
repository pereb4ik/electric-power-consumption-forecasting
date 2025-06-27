import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error    


def show_statistics(model, X_test, X_train, y_test, y_train):
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    plt.plot(y_test, label="original", ls='--')
    plt.plot(y_pred, label="pred")
    plt.legend()
    
    print("Test MAE = %.4f" % mean_absolute_error(y_test, y_pred))
    print("Train MAE = %.4f" % mean_absolute_error(y_train, y_train_pred))

    print()

    print("Test MAPE = %.4f" % mean_absolute_percentage_error(y_test, y_pred))
    print("Train MAPE = %.4f" % mean_absolute_percentage_error(y_train, y_train_pred))

    print()

    print("r^2 on test data : %f" % r2_score(y_test, y_pred))
    print("Explained_variance on test data : %f" % explained_variance_score(y_test, y_pred))

    print()



def another_show_statistics(y_test, y_pred):
    plt.plot(y_test, label="original", ls='--')
    plt.plot(y_pred, label="pred")
    plt.legend()
    
    print("Test MAE = %.10f" % mean_absolute_error(y_test, y_pred))
    print()

    print("Test MAPE = %.10f" % mean_absolute_percentage_error(y_test, y_pred))
    print()

    print("r^2 on test data : %f" % r2_score(y_test, y_pred))
    print("Explained_variance on test data : %f" % explained_variance_score(y_test, y_pred))

    print()


# train test cut special for time series
# here assumse that day length = 144

def train_test_cut(X_, y_, train_days):
    day_length = 144
    X_train = X_[:day_length * train_days]
    X_test = X_[day_length * train_days:]
    
    y_train = np.array(y_[:day_length * train_days])
    y_test = np.array(y_[day_length * train_days:])
    
    #X_test = X_test.reshape(-1, 1)
    #X_train = X_train.reshape(-1, 1)
    return X_train, X_test, y_train, y_test



# Show dependence next value with specified lag=shift

def shift_plot(XX, shift=1):
    plt.scatter(XX[:-shift].values, XX[shift:].values)
    plt.xlabel("$y_t$")
    plt.ylabel("$y_{t+" + str(shift) + "}$")


# Differentiate series with lag=shift
# Also you can use np.diff for diff with shift=1
def season_diff(XX, shift):
    return (XX[shift:].values - XX[:-shift].values)



def check_residuals_for_autocorr(resid):
    # https://stackoverflow.com/questions/71408558/getting-durbin-watson-figure-from-statsmodels-api
    # https://robjhyndman.com/hyndsight/ljung-box-test/
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    dw = durbin_watson(resid)
    print(f"Durbin-Watson: {dw}")
    
    print()
    
    lb = acorr_ljungbox(resid, boxpierce=True)
    print("The Ljung-Box test:")
    print(lb)
    
    print()
    print("So, if no autocorrelation in res, p-val > 0.05 and Durbin-Watson = 2")



def adf_test(timeseries):
    # https://www.geeksforgeeks.org/how-to-check-if-time-series-data-is-stationary-with-python/
    # https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
    
    # import the adfuller function from statsmodel 
    # # package to perform ADF test
    from statsmodels.tsa.stattools import adfuller
    
    # passing the extracted passengers count to adfuller function.
    # # result of adfuller function is stored in a res variable
    res = adfuller(timeseries, autolag="AIC")
    
    # Printing the statistical result of the adfuller test
    print('Augmneted Dickey-Fuller Statistic: %f' % res[0])
    print('p-value: %f' % res[1])
    
    # printing the critical values at different alpha levels.
    print('critical values at different levels:')
    for k, v in res[4].items():
        print('\t%s: %.6f' % (k, v))
    
    print()
    print("If p-value <= significance level (default: 0.05) or ADF statistic < critical value - there is stationarity")




def kpss_test(timeseries):
    from statsmodels.tsa.stattools import kpss
    print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="c", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output.round(5))
    print()
    print("If p-val < 0.05, then series is non-stationary")





def analysis_of_residuals(model, X_test, y_test):
    # https://scikit-learn.ru/stable/modules/model_evaluation.html#visualization-regression-evaluation
     
    from sklearn.metrics import PredictionErrorDisplay
    
    f, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    
    y_pred = model.predict(X_test)

    PredictionErrorDisplay.from_predictions(
        y_test,
        y_pred,
        kind="actual_vs_predicted",
        ax=ax0,
        scatter_kwargs={"alpha": 0.5},
    )

    PredictionErrorDisplay.from_predictions(
        y_test,
        y_pred,
        kind="residual_vs_predicted",
        ax=ax1,
        scatter_kwargs={"alpha": 0.5},
    )
    
    ax0.set_title("Actual vs. Predicted values")
    ax1.set_title("Residuals vs. Predicted Values")



    # Here start check for autocorrelation
    resid = y_test - y_pred
    check_residuals_for_autocorr(resid)


    import statsmodels.api as sm
    
    sm.graphics.tsa.plot_pacf(resid, lags=40)
    plt.show()


    sm.graphics.tsa.plot_acf(resid, lags=40)
    plt.show()


    print("Check for bias=")
    print(resid.mean())
    print()

    # Here start check for stationarity

    adf_test(resid)

    print()
    print()

    kpss_test(resid)