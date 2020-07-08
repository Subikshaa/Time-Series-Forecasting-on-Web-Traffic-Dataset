# Time-Series-Forecasting-on-Web-Traffic-Dataset

Performed time-series analysis and forecasting on Google's web traffic dataset to forecast number of views of Wikipedia web pages. This can help Google take effective measures to handle the web traffic.

**Dataset:** Kaggle Web Traffic Time Series Forecasting (https://www.kaggle.com/c/web-traffic-time-series-forecasting)

**Technologies and libraries used:** Python, AWS EC2, AWS S3, Flask, Statsmodels, Prophet, Tensorflow, Matplotlib.

![](Demo.gif)

**Key Highlights:**
- Performed time series analysis, anomaly detection using `Isolation Forest` and interpolation using `rolling mean`
- Explored various time series forecasting models including ARMA, ARIMA, Exponential Smoothing, Prophet, CNN and LSTM and compared performance using `RMSE`
- Developed `flask` app to render forecast plots generated using saved models
- Practiced fetching data from `AWS S3` using boto3 and deployed the flask app on `AWS EC2` instance using nginx and gunicorn
