import os
import pickle
from flask import Flask, request, jsonify, render_template
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app = Flask(__name__)
SECRET_KEY = os.urandom(24)

@app.route('/')
@app.route('/result', methods=['POST'])
def home():
    fcast = ""
    if request.method == 'POST':
        method = request.form['forecast']

        if method == "arma":
            model = pickle.load(open('arma_model.pkl', 'rb'))
            result = model.fit()
            forecast_values = result.get_forecast(steps=30)
            forecast_values_mean = forecast_values.predicted_mean
            fig, ax = plt.subplots(figsize=(15,4))
            plt.plot(forecast_values_mean)
            plt.savefig('static/plot.png')

        return render_template('index.html', forecast=method, fcast='/static/plot.png')
    return render_template('index.html', fcast=fcast)

if __name__ == '__main__':
    app.secret_key = SECRET_KEY
    app.run(debug=True)