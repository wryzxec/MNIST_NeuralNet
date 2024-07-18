import numpy as np
import struct
from array import array
from os.path import join
from pathlib import Path

class MnistDataHandler: 
    def __init__(self):

        self.current_directory = Path(__file__).resolve().parent
        self.training_images_filepath = join(self.current_directory, 'mnist/train-images-idx3-ubyte/train-images-idx3-ubyte')
        self.training_labels_filepath = join(self.current_directory, 'mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte')
        self.test_images_filepath = join(self.current_directory, 'mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
        self.test_labels_filepath = join(self.current_directory, 'mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            images[i][:] = img            
        
        return images, np.array(labels)
    
    def normalise(self, x):
        x = x / 255
        return x

    def load_training_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_train = np.reshape(x_train, (-1, 28*28))
        
        return x_train, y_train
    
    def load_test_data(self):
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        x_test = np.reshape(x_test, (-1, 28*28))

        return x_test, y_test
    
    def create_mini_batches(self, X, Y, batch_size):
        mini_batches = []

        data = np.vstack((X, Y))
        
        indices_permutation = np.random.permutation(data.shape[1])
        data = data[:, indices_permutation]

        n_minibatches = data.shape[1] // batch_size

        for i in range(n_minibatches):
            mini_batch = data[:, i * batch_size : (i + 1) * batch_size]
            X_mini = mini_batch[:-1, :]
            Y_mini = mini_batch[-1, :]

            mini_batches.append((X_mini, Y_mini))

        if data.shape[1] % batch_size != 0:
            mini_batch = data[:, n_minibatches * batch_size : data.shape[1]]
            X_mini = mini_batch[:-1, :]
            Y_mini = mini_batch[-1, :]

            mini_batches.append((X_mini, Y_mini))

        return mini_batches