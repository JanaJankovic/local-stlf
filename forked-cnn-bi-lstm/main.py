
from pandas import read_csv
from keras.callbacks import CSVLogger
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import RepeatVector
from keras.layers import Dropout
from util import convert_train_val, split_dataset, MetricsLogger
from sklearn.preprocessing import MinMaxScaler


def build_and_train_model_CNNBiLSTM(train, val, n_input, horizon, epochs, batch_size, scaler):
	train_x, train_y = convert_train_val(train, n_input, horizon)
	val_x, val_y = convert_train_val(val, n_input, horizon)
	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
	val_y = val_y.reshape((val_y.shape[0], val_y.shape[1], 1))

	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

	model = Sequential()
	model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
	model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(RepeatVector(n_outputs))
	model.add(Bidirectional(LSTM(100, activation='relu', return_sequences=False)))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(horizon))
	model.compile(loss='mse', optimizer='adam')
	print(model.summary())

	# Add custom logging callback
	log_callback = MetricsLogger(
		train_data=(train_x, train_y),
		val_data=(val_x, val_y),
		scaler=scaler,
		eval_log_path="logs/training_eval.csv",
		loss_log_path="logs/training_log.csv",
		model_save_dir="models"
	)

	hist = model.fit(train_x, train_y,
					 validation_data=(val_x, val_y),
					 epochs=epochs,
					 batch_size=batch_size,
					 verbose=1,
					 callbacks=[log_callback])
	return model, hist


if __name__ == "__main__":
	dataset = read_csv('data/data.csv', header=0, 
                   infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])

	n_input = 24 * 14
	horizon = 1
	batch_size = 64
	epochs = 20

	# 1. Split raw dataset first (before scaling)
	train, val, test = split_dataset(dataset, split_ratios=(0.6, 0.1, 0.3), daily_step=n_input, start=0)
	train = train.to_numpy()
	val = val.to_numpy()
	test = test.to_numpy()

	# 2. Flatten training set to fit scaler (reshape to 2D: [samples * timesteps, features])
	train_2d = train.reshape(-1, train.shape[-1])
	scaler = MinMaxScaler()
	train_scaled = scaler.fit_transform(train_2d)
	train = train_scaled.reshape(train.shape)

	# 3. Scale val and test using the same scaler (transform only)
	val = scaler.transform(val.reshape(-1, val.shape[-1])).reshape(val.shape)
	test = scaler.transform(test.reshape(-1, test.shape[-1])).reshape(test.shape)

	# 5. Confirm shapes
	print('train size: ', train.shape)
	print('valid size: ', val.shape)
	print('test size : ', test.shape)

	build_and_train_model_CNNBiLSTM(train, val, n_input, horizon, epochs, batch_size, scaler)


