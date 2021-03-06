import datetime  # add to requirements

import tensorflow as tf  # add to requirements
import yfinance as yf
from flask import Flask
from flask import (request, jsonify)
from pandas import DataFrame
from pandas import concat
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)


def get_stock_price(stock_name):
    end = datetime.datetime.today() - datetime.timedelta(days=1)
    days = 0
    start = end - datetime.timedelta(days=days)
    df = yf.download(stock_name, start, end)
    data = df['Adj Close'].values
    return data.tolist()[0]


def exchange(base_price, to_stock):
    to_price = get_stock_price(to_stock)
    return "{:.5f} ".format((base_price / to_price)) + to_stock.split("-")[0]


def predict(stock_name, type, number_of_days):
    end = datetime.datetime.today() - datetime.timedelta(days=1)
    days = number_of_days if type == 'past_data' else 285
    start = end - datetime.timedelta(days=days)

    df = yf.download(stock_name, start, end)

    day_list = []
    try:
        for d in range(5):
            day_list.append(df.index[-1] + datetime.timedelta(days=d + 1))
    except:
        return {}

    n_input = 5
    n_nodes = [100, 50, 25, 15, 10]
    n_epochs = 300
    n_batch = 30
    num_step = number_of_days
    n_test = 1

    n_in = 5
    n_out = num_step

    data = df['Adj Close'].values

    if type == 'past_data':
        return {'stock_name': stock_name,
                'predictions': [],
                'past_data': data.tolist()}

    def series_to_supervised(data, n_in, n_out=1):
        df = DataFrame(data)
        cols = list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
        # put it all together
        agg = concat(cols, axis=1)
        # drop rows with NaN values
        agg.dropna(inplace=True)
        return agg.values

    prepared_data = series_to_supervised(data, n_in, n_out)
    train_x, train_y = prepared_data[:, :-num_step], prepared_data[:, -num_step:]
    X_train, X_test = train_x[:-n_test, :], train_x[-n_test:, :]
    y_train, y_test = train_y[:-n_test], train_y[-n_test:]

    model = tf.keras.Sequential([tf.keras.layers.Dense(n_nodes[0], activation='relu', input_dim=n_input),
                                 tf.keras.layers.Dense(n_nodes[1]),
                                 tf.keras.layers.Dense(n_nodes[2]),
                                 tf.keras.layers.Dense(n_nodes[3]),
                                 tf.keras.layers.Dense(n_nodes[4]),
                                 tf.keras.layers.Dense(num_step)
                                 ])
    model.compile(loss='mse', optimizer='adam')
    # fit model
    model.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch, verbose=0)
    my_predictions = model.predict(X_test)

    return {'stock_name': stock_name,
            'predictions': my_predictions[0].tolist(),
            'past_data': []}


@app.route('/predict', methods=['POST'])
def predict_api():
    if request.method == 'POST':
        response = {}
        type = request.form['type']
        no_of_days = int(request.form['noOfDays'])
        for stock_name in str.split(request.form['currencies'], ","):
            response[stock_name] = predict(stock_name, type, no_of_days)
    return jsonify(response)


@app.route('/exchange', methods=['POST'])
def exchange_api():
    from_currencies = str.split(request.form['from_currencies'], ",")
    to_currencies = str.split(request.form['to_currencies'], ",")
    currencies = np.array(from_currencies + to_currencies)

    response = {'base_price': {}, 'exchange': {}}
    for c in np.unique(currencies):
        response['base_price'][c] = "$ {:.5f}".format(get_stock_price(c))

    for base in from_currencies:
        base_price = get_stock_price(base)
        for stock_name in str.split(request.form['to_currencies'], ","):
            response['exchange'][base + ' -> ' + stock_name] = "1 " + base.split("-")[0] + " = " + exchange(base_price,
                                                                                                            stock_name)
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
