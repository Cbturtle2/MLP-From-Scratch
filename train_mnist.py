import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import struct
from mlp import MultilayerPerceptron, Layer, Relu, Sigmoid, Softmax, CrossEntropy
import numpy as np
from array import array
from os.path import join

# Function to download and extract MNIST data


def load_mnist():

    # MNIST Data Loader Class
    class MnistDataloader(object):
        def __init__(self, training_images_filepath, training_labels_filepath,
                     test_images_filepath, test_labels_filepath):
            self.training_images_filepath = training_images_filepath
            self.training_labels_filepath = training_labels_filepath
            self.test_images_filepath = test_images_filepath
            self.test_labels_filepath = test_labels_filepath

        def read_images_labels(self, images_filepath, labels_filepath):
            # Read labels
            with open(labels_filepath, 'rb') as file:
                magic, size = struct.unpack(">II", file.read(8))
                if magic != 2049:
                    raise ValueError(
                        'Magic number mismatch, expected 2049, got {}'.format(magic))
                labels = array("B", file.read())

            # Read images
            with open(images_filepath, 'rb') as file:
                magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
                if magic != 2051:
                    raise ValueError(
                        'Magic number mismatch, expected 2051, got {}'.format(magic))
                image_data = array("B", file.read())
            images = []
            for i in range(size):
                # Convert flat image data to a 28x28 array
                img = np.array(
                    image_data[i * rows * cols:(i + 1) * rows * cols])
                img = img.reshape(rows, cols)
                images.append(img)
            return images, list(labels)

        def load_data(self):
            x_train, y_train = self.read_images_labels(self.training_images_filepath,
                                                       self.training_labels_filepath)
            x_test, y_test = self.read_images_labels(self.test_images_filepath,
                                                     self.test_labels_filepath)
            return (x_train, y_train), (x_test, y_test)

    # Set file paths (adjust these paths as needed)
    input_path = './input'
    training_images_filepath = join(
        input_path, 'train-images-idx3-ubyte', 'train-images-idx3-ubyte')
    training_labels_filepath = join(
        input_path, 'train-labels-idx1-ubyte', 'train-labels-idx1-ubyte')
    test_images_filepath = join(
        input_path, 't10k-images-idx3-ubyte', 't10k-images-idx3-ubyte')
    test_labels_filepath = join(
        input_path, 't10k-labels-idx1-ubyte', 't10k-labels-idx1-ubyte')

    # Load data using the MnistDataloader
    loader = MnistDataloader(training_images_filepath, training_labels_filepath,
                             test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = loader.load_data()

    # Convert lists to  numpy arrays and flatten each image (28x28 -> 784)
    train_images = np.array([img.reshape(-1) for img in x_train])
    test_images = np.array([img.reshape(-1) for img in x_test])
    train_labels = np.array(y_train)
    test_labels = np.array(y_test)

    return train_images, train_labels, test_images, test_labels


# Load MNIST data
train_images, train_labels, test_images, test_labels = load_mnist()

# Normalize pixel values to [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# One-hot encode the labels


def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]


train_labels_one_hot = one_hot_encode(train_labels)
test_labels_one_hot = one_hot_encode(test_labels)

# Split training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    train_images, train_labels_one_hot, test_size=0.2, random_state=42
)

input_dim = 28 * 28
hidden_dim1 = 64
hidden_dim2 = 32
output_dim = 10

layers = [
    Layer(input_dim, hidden_dim1, Relu()),
    Layer(hidden_dim1, hidden_dim2, Relu()),
    Layer(hidden_dim2, output_dim, Softmax())
]

model = MultilayerPerceptron(layers)


# Train
loss_func = CrossEntropy()
learning_rate = 0.01
batch_size = 128
epochs = 20

training_losses, validation_losses = model.train(
    X_train, y_train, X_val, y_val,
    loss_func, learning_rate, batch_size, epochs
)

# Evaluate
y_pred_prob = model.forward(test_images)
y_pred = np.argmax(y_pred_prob, axis=1)
accuracy = np.mean(y_pred == test_labels)
print(f"Test Accuracy: {accuracy:.4f}")

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), training_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), validation_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('mnist_loss_curves.png')
plt.show()

plt.figure(figsize=(15, 8))
for digit in range(10):
    idx = np.where(test_labels == digit)[0][0]

    pred = y_pred[idx]

    plt.subplot(2, 5, digit + 1)
    plt.imshow(test_images[idx].reshape(28, 28), cmap='gray')
    plt.title(f"True: {test_labels[idx]}, Pred: {pred}")
    plt.axis('off')

plt.tight_layout()
plt.savefig('mnist_predictions.png')
plt.show()
