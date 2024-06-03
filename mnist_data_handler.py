import numpy as np
import struct
from array import array
from os.path import join

import random
import matplotlib.pyplot as plt
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
    
    def plot_images(self, images, title_texts):

        images = np.reshape(images, (np.shape(images)[0], 28, 28))

        cols = 5
        rows = int(len(images)/cols) + 1 

        images_plot = plt.figure(figsize=(5,5))
        images_plot.subplots_adjust(hspace = 1)

        index = 1    
        for i in range(len(images)):        
            image = images[i]        
            title_text = title_texts[i]
            image_subplot = plt.subplot(rows, cols, index)
            image_subplot.axis('off')
            image_subplot.imshow(image, cmap=plt.cm.gray)
            
            if (title_text != ''):
                image_subplot.set_title(title_text, fontsize = 15)
            
            images_plot.add_subplot(image_subplot)
            index += 1
        
        return images_plot
    
    def generate_random_training_samples(self, sample_count, x_train, y_train):
        
        images_to_show = []
        titles_to_show = []
        for i in range(0, sample_count):
            r = random.randint(1, 60000)
            images_to_show.append(x_train[r])
            titles_to_show.append('y = ' + str(y_train[r]))

        return images_to_show, titles_to_show

def main(): 
    mnist_data_handler = MnistDataHandler()
    x_train, y_train = mnist_data_handler.load_training_data()
    images_to_show, titles_to_show = mnist_data_handler.generate_random_training_samples(20, x_train, y_train)
    mnist_data_handler.plot_images(images_to_show, titles_to_show)

    plt.show()

if __name__ == "__main__":
    main()