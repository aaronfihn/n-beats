import warnings

import numpy as np

from nbeats_keras.model import NBeatsNet as NBeatsKeras
# from nbeats_pytorch.model import NBeatsNet as NBeatsPytorch

warnings.filterwarnings(action='ignore', message='Setting attributes')


def get_data() -> (np.array, np.array, np.array, np.array):
    full_ts = np.loadtxt('run-of-river_production_load.csv',
                   delimiter=',',
                   skiprows=1,
                   usecols=1)

    # Split into training time series (including training and validation subsets) and testing time series
    validation_frac = 0.1
    testing_frac = 0.1
    validation_length = round(len(full_ts) * validation_frac)
    testing_length = round(len(full_ts) * testing_frac)
    training_subset_length = len(full_ts) - validation_length - testing_length
    training_subset_ts = full_ts[:training_subset_length]
    validation_ts = full_ts[training_subset_length:training_subset_length+validation_length]
    testing_ts = full_ts[training_subset_length+validation_length:]

    # Bootstrap the training subset into 45k training samples
    horizon = 4 * 24 * 2  # 4 samples/hour * 24 hours/day * 2 days
    num_train_samples = 45_000
    lookback_multipliers = list(range(2, 8))
    window_length = (max(lookback_multipliers) + 1) * horizon
    rng = np.random.default_rng(seed=42)
    subtrain_start_idx = rng.choice(range(training_subset_length - window_length), num_train_samples)  # index to start training sample
    x_subtrain, y_subtrain = [], []
    for idx in subtrain_start_idx:
        x_subtrain.append(training_subset_ts[idx: idx + window_length - horizon])
        y_subtrain.append(training_subset_ts[idx + window_length - horizon: idx + window_length])
    x_subtrain = np.array(x_subtrain)
    y_subtrain = np.array(y_subtrain)

    # Bootstrap the validation subset into 5k test
    num_val_samples = 5_000
    val_start_idx = rng.choice(range(validation_length - window_length), num_val_samples)
    x_val, y_val = [], []
    for idx in val_start_idx:
        x_val.append(validation_ts[idx: idx + window_length - horizon])
        y_val.append(validation_ts[idx + window_length - horizon: idx + window_length])
    x_val = np.array(x_val)
    y_val = np.array(y_val)

    return x_subtrain, y_subtrain, x_val, y_val


def main():
    # https://keras.io/layers/recurrent/
    # At the moment only Keras supports input_dim > 1. In the original paper, input_dim=1.
    num_samples, time_steps, input_dim, output_dim = 50_000, 10, 1, 1

    # This example is for both Keras and Pytorch. In practice, choose the one you prefer.
    # for BackendType in [NBeatsKeras, NBeatsPytorch]:
    for BackendType in [NBeatsKeras]:
        # NOTE: If you choose the Keras backend with input_dim>1, you have 
        # to set the value here too (in the constructor).
        backend = BackendType(
            backcast_length=time_steps, forecast_length=output_dim,
            stack_types=(NBeatsKeras.GENERIC_BLOCK, NBeatsKeras.GENERIC_BLOCK),
            nb_blocks_per_stack=2, thetas_dim=(4, 4), share_weights_in_stack=True,
            hidden_layer_units=64
        )

        # Definition of the objective function and the optimizer.
        backend.compile(loss='mae', optimizer='adam')

        # Definition of the data. The problem to solve is to find f such as | f(x) - y | -> 0.
        # where f = np.mean.
        x = np.random.uniform(size=(num_samples, time_steps, input_dim))
        y = np.mean(x, axis=1, keepdims=True)

        # Split data into training and testing datasets.
        c = num_samples // 10
        x_train, y_train, x_test, y_test = x[c:], y[c:], x[:c], y[:c]
        test_size = len(x_test)

        # Train the model.
        print('Training...')
        backend.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=128)

        # Save the model for later.
        backend.save('n_beats_model.h5')

        # Predict on the testing set (forecast).
        predictions_forecast = backend.predict(x_test)
        np.testing.assert_equal(predictions_forecast.shape, (test_size, backend.forecast_length, output_dim))

        # Predict on the testing set (backcast).
        predictions_backcast = backend.predict(x_test, return_backcast=True)
        np.testing.assert_equal(predictions_backcast.shape, (test_size, backend.backcast_length, output_dim))

        # Load the model.
        model_2 = BackendType.load('n_beats_model.h5')

        np.testing.assert_almost_equal(predictions_forecast, model_2.predict(x_test))


if __name__ == '__main__':
    get_data()
