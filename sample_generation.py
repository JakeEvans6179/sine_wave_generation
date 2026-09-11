import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

'''
Use NN to predict a sine wave.

Generate inputs and outputs and get NN to predict 
'''


x_values = np.linspace(0, 2*np.pi, 1000)
y_values = np.zeros(1000)
print(x_values.shape)


for i in range(1000):

    sine_value = np.sin(x_values[i])

    y_values[i] = sine_value


plt.plot(x_values, y_values)
#plt.show()




'''
x_test = np.array([1,2,3])

print(x_test)

x_test = np.reshape(x_test,(-1,1)) #reshape to 1 column and as many rows to fit the data 
#e.g. [1,2,3] --> [[1],
#                  [2],
#                  [3]]
'''

#reshape the x and y arrays 
x_values = np.reshape(x_values, (-1,1))
y_values = np.reshape(y_values, (-1,1))

print(x_values.shape) #expected shape (1000,1)
print(y_values.shape) #expected shape (1000,1)

#split data 

x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size = 0.1, random_state = 69)

print(f"x_train shape: {x_train.shape}")
print(f"y-train shape: {y_train.shape}")

print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

input_shape = (x_train.shape[1],) #input_shape expects a tuple argument e.g. in this case 1 input value (1,)
print(input_shape)

model = Sequential([
    Input(shape = input_shape),
    Dense(units = 16, activation = 'relu', ),
    Dense(units = 8, activation = 'relu'),

    Dense(units = 1, activation = 'linear') #output value
])

model.summary()

model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

es = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )

history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=1000,
        batch_size=64,
        verbose=1,
        callbacks=[es]
    )

# history.history is the actual dictionary
metrics = history.history

print(f"Final Training MSE: {metrics['loss'][-1]:.4f}")
print(f"Final Validation MSE: {metrics['val_loss'][-1]:.4f}")

print(f"Final Training RMSE: {metrics['root_mean_squared_error'][-1]:.4f}")
print(f"Final Validation RMSE: {metrics['val_root_mean_squared_error'][-1]:.4f}")

plt.figure()
plt.plot(metrics['root_mean_squared_error'], label='Training RMSE')
plt.plot(metrics['val_root_mean_squared_error'], label='Validation RMSE')
plt.title('Model RMSE During Training')
plt.ylabel('RMSE')
plt.xlabel('Epoch')
plt.legend()
plt.show()



print("Evaluating on Test Set...")
results = model.evaluate(x_test, y_test, verbose=1)

# The results list matches the order from model.compile()
# Index 0 is always the Loss. The rest are your chosen metrics.
test_mse = results[0]
test_rmse = results[1]

print(f"\n--- Final Test Scores ---")
print(f"Test MSE:  {test_mse:.5f}")
print(f"Test RMSE: {test_rmse:.5f}")


#running test 

inference_test_values = [0.1, 0.5, 1, 2, 5]

inference_test_values = np.array([0.1, 0.5, 1, 2, 5]).reshape(-1, 1)

# 2. Predict ALL values at exactly the same time (Batch Prediction)
predicted_sine_values = model.predict(inference_test_values)

# 3. Calculate all actual values at the same time
actual_sine_values = np.sin(inference_test_values)

print(actual_sine_values)
print(predicted_sine_values)

model.save('sine_prediction.keras')
print("saved model")


#convert to tflite 

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] #by default quantises weights to int8 while keeping input and output as floats


tflite_quant_model = converter.convert()


with open('sine_model_weight_quant.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print("Weight-only quantized model saved")