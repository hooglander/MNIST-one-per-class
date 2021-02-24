import tensorflow as tf
import numpy as np
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from scipy.ndimage.interpolation import shift, rotate
from matplotlib import pyplot as plt
from numpy.random import permutation

def main(handpicked=True, epochs=250):
    rotatable=[0, 1, 6, 8, 9]
    mirrorable=[0, 1, 8]
    pixel_shifts=((1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1))
    input_shape=(28, 28, 1)
    
    (train_images, train_labels), (test_images, test_labels)=tf.keras.datasets.mnist.load_data()
    train_set=(train_images, train_labels)

    if (handpicked):
        new_train_set_indices=[952, 951, 220, 50, 945, 1071, 66, 258, 972, 264]
    else:
        classes=np.zeros(10).astype(int)
        new_train_set_indices=[]
        while sum(classes)!=10:
            for i in permutation(len(train_labels)):
                if classes[train_set[1][i]]==0:
                    classes[train_set[1][i]]=1
                    new_train_set_indices.append(i)
    
    new_train_images=train_images[new_train_set_indices]
    new_train_images=new_train_images.reshape(new_train_images.shape[0], new_train_images.shape[1], new_train_images.shape[2], 1)
    new_train_labels=train_labels[new_train_set_indices]

    test_images=test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 1)
    
    new_train_images_shifted=np.copy(new_train_images)
    new_train_labels_shifted=np.copy(new_train_labels)

    for dx, dy in pixel_shifts:
         for image, label in zip(new_train_images, new_train_labels):
                new_train_images_shifted=np.append(new_train_images_shifted, shift_image(image, dx, dy), axis=0)
                new_train_labels_shifted=np.append(new_train_labels_shifted, label)

    new_train_images_rotated=np.copy(new_train_images_shifted)
    new_train_labels_rotated=np.copy(new_train_labels_shifted)

    for image, label in zip(new_train_images_shifted, new_train_labels_shifted):
        if label in rotatable:
                new_train_images_rotated=np.append(new_train_images_rotated, rotate_image(image), axis=0)
                if label==6:
                    new_train_labels_rotated=np.append(new_train_labels_rotated, 9)
                elif label==9:
                    new_train_labels_rotated=np.append(new_train_labels_rotated, 6)
                else:
                    new_train_labels_rotated=np.append(new_train_labels_rotated, label)

    new_train_images_mirrored=np.copy(new_train_images_rotated)
    new_train_labels_mirrored=np.copy(new_train_labels_rotated)

    for image, label in zip(new_train_images_rotated, new_train_labels_rotated):
        if label in mirrorable:
                new_train_images_mirrored=np.append(new_train_images_mirrored, mirror_image(image), axis=0)
                new_train_labels_mirrored=np.append(new_train_labels_mirrored, label)
    
    new_train_images_mirrored=new_train_images_mirrored.astype('float32')
    test_images=test_images.astype('float32')

    new_train_images_mirrored/=255
    test_images/=255

    model=Sequential()
    model.add(Conv2D(28, kernel_size=(8, 8), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(6, 6)))
    model.add(Flatten())
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation =tf.nn.softmax))
    
    model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    history=model.fit(x=new_train_images_mirrored, y=new_train_labels_mirrored, epochs=epochs, validation_split=0.2)
    
    print('\nTest loss and accuracy:')
    model.evaluate(test_images, test_labels)

    save_plot(history.history['accuracy'], history.history['val_accuracy'], 'model accuracy', 'accuracy')
    save_plot(history.history['loss'], history.history['val_loss'], 'model loss', 'loss')

def shift_image(image, dx, dy):
    image=image.reshape((28, 28))
    shifted_image=shift(image, [dy, dx], cval=0, mode="constant")

    return shifted_image.reshape((1, 28, 28, 1))

def rotate_image(image):
    image=image.reshape((28, 28))
    rotated_image=rotate(image, angle=180, cval=0, mode="constant")
    
    return rotated_image.reshape((1, 28, 28, 1))

def mirror_image(image):
    image=image.reshape((28, 28))
    mirrored_image=np.flip(image, axis=1)
        
    return mirrored_image.reshape((1, 28, 28, 1))

def save_plot(plot_1, plot_2, title, y_label):
    plt.clf()
    plt.plot(plot_1)
    plt.plot(plot_2)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    plt.savefig(y_label, dpi=300)

parser=argparse.ArgumentParser()
parser.add_argument('-H', '--handpicked', type=bool, metavar='', help='Use a handpicked set of training data')
parser.add_argument('-e', '--epochs', type=int, metavar='', help='Default value is 250')
args=parser.parse_args()

if __name__=='__main__':
    if args.handpicked==None:
        args.handpicked=True

    if args.epochs==None or args.epochs<1:
        args.epochs=250

    main(args.handpicked, args.epochs)