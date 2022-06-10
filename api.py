import datetime  # add to requirements

import tensorflow as tf  # add to requirements
import yfinance as yf
from flask import Flask
from flask import (request, jsonify)
from pandas import DataFrame
from pandas import concat
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict_api():
    if request.method == 'POST':
        stock_name = request.form['name']
        end = datetime.datetime.today() - datetime.timedelta(days=1)
        start = end - datetime.timedelta(days=285)

        df = yf.download(stock_name, start, end)

        day_list = []
        try:
            for d in range(5):
                day_list.append(df.index[-1] + datetime.timedelta(days=d + 1))
        except:
            return jsonify({}), 404

        n_input = 5
        n_nodes = [100, 50, 25, 15, 10]
        n_epochs = 300
        n_batch = 30
        num_step = 5
        n_test = 1

        n_in = 5
        n_out = num_step

        data = df['Adj Close'].values

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
        starting_year = str(day_list[0].year)
        starting_month = str(day_list[0].month)
        starting_day = str(day_list[0].day)
        first_day = starting_month + '-' + starting_day + '-' + starting_year

    return jsonify({'predictions': my_predictions[0].tolist(),
                    'prediction1': float(my_predictions[0][0]),
                    'prediction2': float(my_predictions[0][1]), 'prediction3': float(my_predictions[0][2]),
                    'prediction4': float(my_predictions[0][3]), 'prediction5': float(my_predictions[0][4]),
                    'starting_day': first_day, 'stock_name': stock_name, 'past_data': data.tolist()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
