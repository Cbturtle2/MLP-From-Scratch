import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import requests
from io import StringIO
from mlp import MultilayerPerceptron, Layer, Relu, Sigmoid, Tanh, Linear, SquaredError

# Download the Vehicle MPG dataset


def get_mpg_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    response = requests.get(url)
    data_str = response.text

    # The dataset uses fixed-width formatting, convert to CSV format
    column_names = ['mpg', 'cylinders', 'displacement', 'horsepower',
                    'weight', 'acceleration', 'model_year', 'origin', 'car_name']

    # Read data with pandas
    data = pd.read_csv(StringIO(data_str), delim_whitespace=True,
                       header=None, names=column_names)

    # Drop the car name column
    data = data.drop('car_name', axis=1)

    data = data.replace('?', np.nan)
    data = data.apply(pd.to_numeric, errors='ignore')

    # Drop rows with missing values or fill them
    data = data.dropna()

    # Extract features and target
    X = data.drop('mpg', axis=1).values
    y = data['mpg'].values.reshape(-1, 1)

    return X, y


# Get the data
X, y = get_mpg_data()

# Split the data
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.15/0.85, random_state=42)


input_dim = X_train.shape[1]
output_dim = 1

layers = [
    Layer(input_dim, 24, Relu()),
    Layer(24, 12, Relu()),
    Layer(12, output_dim, Linear())
]

model = MultilayerPerceptron(layers)

# Train
loss_func = SquaredError()
learning_rate = 0.001
batch_size = 16
epochs = 200

training_losses, validation_losses = model.train(
    X_train, y_train, X_val, y_val,
    loss_func, learning_rate, batch_size, epochs,
)

y_pred = model.forward(X_test)
test_loss = loss_func.loss(y_test, y_pred)
test_loss_mean = np.mean(test_loss)
print(f"Test Loss: {test_loss_mean:.4f}")


plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), training_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), validation_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('mpg.png')
plt.show()


print("Predicted vs True MPG for 10 samples:")
print(pd.DataFrame(
    {'Predicted MPG': y_pred[0:10].flatten(), 'True MPG': y_test[0:10].flatten()}))
