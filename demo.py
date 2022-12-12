import tensorflow as tf
import warnings

import numpy as np

from nbeats_keras.model import NBeatsNet as NBeatsKeras

warnings.filterwarnings(action='ignore', message='Setting attributes')


class MASE(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        sigma_num = tf.math.abs(y_true - y_pred)
        sigma_denom = tf.math.abs(y_true)
        sigma_frac = tf.math.divide_no_nan(sigma_num, sigma_denom)
        return 100 * tf.math.reduce_mean(sigma_frac, axis=-1)


class SMAPE(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        sigma_num = tf.math.abs(y_true - y_pred)
        sigma_denom = tf.math.abs(y_true) + tf.math.abs(y_pred)
        sigma_frac = tf.math.divide_no_nan(sigma_num, sigma_denom)
        return 200 * tf.math.reduce_mean(sigma_frac, axis=-1)


def get_data() -> (np.array, np.array, np.array, np.array):
    full_ts = np.loadtxt('run-of-river_production_load.csv',
                   delimiter=',',
                   skiprows=1,
                   usecols=1)

    # Split into training TS (including training and validation subsets) and testing TS
    validation_frac = testing_frac = 0.1
    validation_length = round(len(full_ts) * validation_frac)
    testing_length = round(len(full_ts) * testing_frac)
    training_subset_length = len(full_ts) - validation_length - testing_length
    training_subset_ts = full_ts[:training_subset_length]
    validation_ts = full_ts[training_subset_length:training_subset_length+validation_length]
    testing_ts = full_ts[training_subset_length+validation_length:]

    # Normalize all time series
    norm_layer = tf.keras.layers.Normalization(axis=None)
    norm_layer.adapt(training_subset_ts)
    training_subset_ts = norm_layer(training_subset_ts)
    validation_ts = norm_layer(validation_ts)
    testing_ts = norm_layer(testing_ts)

    # Bootstrap the training subset into 50k training samples
    horizon = 4 * 24 * 2  # 4 samples/hour * 24 hours/day * 2 days
    num_train_samples = 50_000
    lookback_multipliers = list(range(2, 8))
    window_length = (max(lookback_multipliers) + 1) * horizon
    rng = np.random.default_rng(seed=42)
    subtrain_start_idx = rng.choice(range(training_subset_length - window_length), num_train_samples)
    x_subtrain, y_subtrain = [], []
    for idx in subtrain_start_idx:
        x_subtrain.append(training_subset_ts[idx: idx + window_length - horizon])
        y_subtrain.append(training_subset_ts[idx + window_length - horizon: idx + window_length])
    x_subtrain = np.array(x_subtrain).reshape(len(x_subtrain), len(x_subtrain[0]), 1)
    y_subtrain = np.array(y_subtrain).reshape(len(y_subtrain), len(y_subtrain[0]), 1)

    # Bootstrap the validation subset into 5k test
    num_val_samples = 5_000
    val_start_idx = rng.choice(range(validation_length - window_length), num_val_samples)
    x_val, y_val = [], []
    for idx in val_start_idx:
        x_val.append(validation_ts[idx: idx + window_length - horizon])
        y_val.append(validation_ts[idx + window_length - horizon: idx + window_length])
    x_val = np.array(x_val).reshape(len(x_val), len(x_val[0]), 1)
    y_val = np.array(y_val).reshape(len(y_val), len(y_val[0]), 1)

    return x_subtrain, y_subtrain, x_val, y_val


def main():
    # Definition of the data. The problem to solve is to find f such as | f(x) - y | -> 0.
    # where f = np.mean.
    # x = np.random.uniform(size=(num_samples, time_steps, input_dim))
    # y = np.mean(x, axis=1, keepdims=True)

    # Split data into training and testing datasets.
    # c = num_samples // 10
    # x_train, y_train, x_test, y_test = x[c:], y[c:], x[:c], y[:c]
    x_train, y_train, x_test, y_test = get_data()

    test_size = len(x_test)

    # https://keras.io/layers/recurrent/
    # At the moment only Keras supports input_dim > 1. In the original paper, input_dim=1.
    num_samples = x_train.shape[0]
    time_steps = x_train.shape[1]
    input_dim = 1
    output_dim = 1

    num_stacks = 10
    hidden_layer_width = 256
    epochs = 100
    batch_size = 256

    stack_types = (NBeatsKeras.GENERIC_BLOCK,) * num_stacks
    thetas_dim = (hidden_layer_width,) * num_stacks

    backend = NBeatsKeras(
        backcast_length=time_steps,
        forecast_length=y_train.shape[1],
        #stack_types=(NBeatsKeras.GENERIC_BLOCK, NBeatsKeras.GENERIC_BLOCK),
        stack_types=stack_types,
        nb_blocks_per_stack=2,
        # thetas_dim=(4, 4),
        thetas_dim=thetas_dim,
        share_weights_in_stack=False,
        hidden_layer_units=hidden_layer_width
    )

    # Definition of the objective function and the optimizer.
    # backend.compile(loss='mase', optimizer='adam')
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    backend.compile(loss=SMAPE(),
                    optimizer=adam_optimizer)

    # Train the model.
    print('Training...')
    backend.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)

    # Save the model for later.
    backend.save('n_beats_model.h5')

    # Predict on the testing set (forecast).
    predictions_forecast = backend.predict(x_test)
    np.testing.assert_equal(predictions_forecast.shape, (test_size, backend.forecast_length, output_dim))

    # Predict on the testing set (backcast).
    predictions_backcast = backend.predict(x_test, return_backcast=True)
    np.testing.assert_equal(predictions_backcast.shape, (test_size, backend.backcast_length, output_dim))

    # Load the model.
    custom_objects = {'SMAPE': SMAPE, 'MASE': MASE}
    model_2 = NBeatsKeras.load('n_beats_model.h5', custom_objects=custom_objects)

    np.testing.assert_almost_equal(predictions_forecast, model_2.predict(x_test))


if __name__ == '__main__':
    main()