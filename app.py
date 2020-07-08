# Import packages
import os
import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd
import boto3
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import time

app = Flask(__name__)
SECRET_KEY = os.urandom(24)

# Read data from S3 bucket
bucket = "flaskforecast"
file_name = "final_data.csv"
s3 = boto3.client('s3',aws_access_key_id = YOUR_ACCESS_KEY,aws_secret_access_key=YOUR_SECRET_KEY)
obj = s3.get_object(Bucket= bucket, Key= file_name)
data_df = pd.read_csv(obj['Body'])

# data_df = pd.read_csv('Preprocessed_data/final_data.csv')

data_df = data_df.set_index('Date')
data_df.index = pd.DatetimeIndex(data_df.index)

# Train, test split
train = data_df[:'2016-09']
test = data_df['2016-10':]


# Display plots based on user's selection of model
@app.route('/')
@app.route('/result', methods=['POST'])
def home():
    fcast = ""
    if request.method == 'POST':
        method = request.form['forecast']

        # Forecasting using saved ARMA model
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
            plt.xlabel('Date')
            plt.ylabel('Views')
            plt.legend(loc='best')
            new_arma_plot = "arma_plot_" + str(time.time()) + ".png"

            for filename in os.listdir('static/'):
                if filename.startswith('arma_plot_'):
                    os.remove('static/' + filename)

            plt.savefig('static/' + new_arma_plot)
            return render_template('index.html', forecast='ARMA', fcast='static/' + new_arma_plot)

        # Forecasting using saved ARIMA model
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
            plt.xlabel('Date')
            plt.ylabel('Views')
            plt.legend(loc='best')
            new_arima_plot = "arima_plot_" + str(time.time()) + ".png"

            for filename in os.listdir('static/'):
                if filename.startswith('arima_plot_'):
                    os.remove('static/' + filename)

            plt.savefig('static/' + new_arima_plot)
            return render_template('index.html', forecast='ARIMA', fcast='static/' + new_arima_plot)

        # Forecasting using saved Exponential Smoothing model
        elif method == 'exp':
            exp_model = pickle.load(open('model/exp_smoothing_model.pkl', 'rb'))
            exp_smoothing_result = exp_model.fit(smoothing_level=0.5,optimized=True)
            test.index=pd.DatetimeIndex(test.index)
            exp_smoothing_forecast = exp_smoothing_result.forecast(test.shape[0])
            exp_smoothing_forecast=exp_smoothing_forecast.reset_index().rename(columns={'index':'Date',0:'Views'}).set_index('Date')

            fig, ax = plt.subplots(figsize=(15,4))
            test.rename(columns={'Views':'Actual data'}).plot(ax=ax,color='blue')
            exp_smoothing_forecast.rename(columns={'Views':'Forecast'}).plot(ax=ax,color='red')
            plt.xlabel('Date')
            plt.ylabel('Views')
            plt.legend(loc='best')

            new_exp_plot = "exp_plot_" + str(time.time()) + ".png"

            for filename in os.listdir('static/'):
                if filename.startswith('exp_plot_'):
                    os.remove('static/' + filename)

            plt.savefig('static/' + new_exp_plot)
            return render_template('index.html', forecast='Exponential Smoothing', fcast='static/' + new_exp_plot)

        # Forecasting using saved Prophet model
        elif method == 'prophet':
            prophet_model = pickle.load(open('model/prophet_model.pkl', 'rb'))
            test.index = pd.DatetimeIndex(test.index)
            future = prophet_model.make_future_dataframe(periods=test.shape[0])
            prophet_forecast = prophet_model.predict(future)

            forecast_prophet = prophet_forecast[['ds','yhat_lower','yhat_upper','yhat']][-test.shape[0]:]
            forecast_prophet = forecast_prophet.set_index('ds')

            fig, ax = plt.subplots(figsize=(15,4))
            pd.plotting.register_matplotlib_converters()
            test.rename(columns={'Views':'Actual data'}).plot(ax=ax,color='blue')
            forecast_prophet.rename(columns={'yhat':'Forecast'})[['Forecast']].plot(ax=ax,color='red')
            plt.fill_between(forecast_prophet.index,forecast_prophet['yhat_lower'],forecast_prophet['yhat_upper'],color='pink',alpha=0.5)
            plt.xlabel('Date')
            plt.ylabel('Views')
            plt.legend(loc='best')

            new_prophet_plot = "prophet_plot_" + str(time.time()) + ".png"

            for filename in os.listdir('static/'):
                if filename.startswith('prophet_plot_'):
                    os.remove('static/' + filename)

            plt.savefig('static/' + new_prophet_plot)
            return render_template('index.html', forecast='Prophet', fcast='static/' + new_prophet_plot)

        # Forecasting using saved AutoARIMA model
        elif method == 'auto_arima':
            auto_arima_result = SARIMAXResults.load('model/auto_arima_model.pkl')
            auto_arima_forecast = auto_arima_result.predict(n_periods=test.shape[0])
            auto_arima_forecast = pd.DataFrame(auto_arima_forecast,index = test.index,columns=['Forecast'])

            fig, ax = plt.subplots(figsize=(15,4))
            test.rename(columns={'Views':'Actual value'}).plot(ax=ax,color='blue')
            auto_arima_forecast[['Forecast']].plot(ax=ax,label='Forecast',color='red')
            ax.set_xlabel('Date')
            ax.set_ylabel('Views')
            plt.legend(loc='best')

            new_auto_arima_plot = "auto_arima_plot_" + str(time.time()) + ".png"

            for filename in os.listdir('static/'):
                if filename.startswith('auto_arima_plot_'):
                    os.remove('static/' + filename)

            plt.savefig('static/' + new_auto_arima_plot)
            return render_template('index.html', forecast='Auto-arima', fcast='static/' + new_auto_arima_plot)
    return render_template('index.html', fcast=fcast)

if __name__ == '__main__':
    app.secret_key = SECRET_KEY
    app.run(debug=True)
