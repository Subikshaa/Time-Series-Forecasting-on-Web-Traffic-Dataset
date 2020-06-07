import os
import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import time

app = Flask(__name__)
SECRET_KEY = os.urandom(24)

data_df = pd.read_csv('Preprocessed_data/final_data.csv')
data_df = data_df.set_index('Date')

train = data_df[:'2016-09']
test = data_df['2016-10':]

@app.route('/')
@app.route('/result', methods=['POST'])
def home():
    fcast = ""
    if request.method == 'POST':
        method = request.form['forecast']

        if method == "arma":
            result=SARIMAXResults.load('model/arma_model.pkl')
            forecast_values = result.get_forecast(steps=test.shape[0])
            forecast_values_mean = forecast_values.predicted_mean
            conf_interval = forecast_values.conf_int()

            arma_forecast_df = pd.DataFrame({'Date':test.index,'Views':forecast_values.predicted_mean,'lower_views':conf_interval['lower Views'].values,'upper_views':conf_interval['upper Views'].values})
            arma_forecast_df = arma_forecast_df.set_index('Date')

            fig, ax = plt.subplots(figsize=(15,4))
            test.rename(columns={'Views':'Actual value'}).plot(ax=ax,color='blue')
            arma_forecast_df[['Views']].rename(columns={'Views':'Forecast'}).plot(ax=ax,label='Forecast',color='red')
            plt.fill_between(arma_forecast_df.index, \
                arma_forecast_df.lower_views, \
                arma_forecast_df.upper_views, \
                color='pink', alpha=0.5)
            plt.ylabel('Views')
            plt.legend(loc='best')
            new_arma_plot = "arma_plot_" + str(time.time()) + ".png"

            for filename in os.listdir('static/'):
                if filename.startswith('arma_plot_'):
                    os.remove('static/' + filename)

            plt.savefig('static/' + new_arma_plot)
            return render_template('index.html', forecast='ARMA', fcast='static/' + new_arma_plot)

        elif method =="arima":
            arima_result = SARIMAXResults.load('model/arima_model.pkl')
            arima_forecast_values = arima_result.get_forecast(steps=test.shape[0])
            arima_forecast_mean = arima_forecast_values.predicted_mean
            arima_conf_interval = arima_forecast_values.conf_int()

            arima_forecast_df = pd.DataFrame({'Date':test.index,'Views':arima_forecast_values.predicted_mean,'lower_views':arima_conf_interval['lower Views'].values,'upper_views':arima_conf_interval['upper Views'].values})
            arima_forecast_df = arima_forecast_df.set_index('Date')

            fig, ax = plt.subplots(figsize=(15,4))
            test.rename(columns={'Views':'Actual value'}).plot(ax=ax,color='blue')
            arima_forecast_df[['Views']].rename(columns={'Views':'Forecast'}).plot(ax=ax,label='Forecast',color='red')
            plt.fill_between(arima_forecast_df.index, \
                arima_forecast_df.lower_views, \
                arima_forecast_df.upper_views, \
                color='pink', alpha=0.5)
            plt.ylabel('Views')
            plt.legend(loc='best')
            new_arima_plot = "arima_plot_" + str(time.time()) + ".png"

            for filename in os.listdir('static/'):
                if filename.startswith('arima_plot_'):
                    os.remove('static/' + filename)

            plt.savefig('static/' + new_arima_plot)
            return render_template('index.html', forecast='ARIMA', fcast='static/' + new_arima_plot)

        elif method == 'exp':
            exp_model = pickle.load(open('model/exp_smoothing_model.pkl', 'rb'))
            exp_smoothing_result = exp_model.fit(smoothing_level=0.5,optimized=True)

            exp_smoothing_forecast = exp_smoothing_result.forecast(test.shape[0])

            plt.plot(data_df,label='Actual data',color='blue')
            plt.plot(exp_smoothing_forecast, label='Forecast',color='red')
            plt.legend(loc='best')

            new_exp_plot = "exp_plot_" + str(time.time()) + ".png"

            for filename in os.listdir('static/'):
                if filename.startswith('exp_plot_'):
                    os.remove('static/' + filename)

            plt.savefig('static/' + new_exp_plot)
            return render_template('index.html', forecast='Exponential Smoothing', fcast='static/' + new_exp_plot)

    return render_template('index.html', fcast=fcast)

if __name__ == '__main__':
    app.secret_key = SECRET_KEY
    app.run(debug=True)
