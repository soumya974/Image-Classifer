import tensorflow
import cv2
import sklearn
import numpy
import os

# Define hyper parameters
learning_rate = 0.002
num_epochs = 100
batch_size = 32
image_height = 64
image_width = 64
num_channels = 3

# Getting the required dataset
def dataset():
    # Load the training data
    train_data = []
    train_labels = []
    for folder in ['data/face', 'data/non face']:
        label = 1 if folder == 'data/face' else 0
        for file in os.listdir(folder):
            image = cv2.imread(os.path.join(folder, file))
            image = cv2.resize(image, (image_width, image_height))
            image = image.astype('float32') / 255.0
            train_data.append(image)
            train_labels.append(label)
    # Load the testing data
    test_data = []
    test_labels = []
    for folder in ['validation/face', 'validation/non face']:
        label = 1 if folder == 'validation/face' else 0
        for file in os.listdir(folder):
            image = cv2.imread(os.path.join(folder, file))
            image = cv2.resize(image, (image_width, image_height))
            image = image.astype('float32') / 255.0
            test_data.append(image)
            test_labels.append(label)
    # Return
    return train_data, train_labels, test_data, test_labels
# Data generator
def data_generator():
    # Convert the data to numpy arrays
    train_data, train_labels, test_data, test_labels = dataset()
    train_data = numpy.array(train_data)
    train_labels = numpy.array(train_labels)
    test_data = numpy.array(test_data)
    test_labels = numpy.array(test_labels)
    # Define data augmentation parameters
    train_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator()
    train_datagen.rotation_range = 20
    train_datagen.width_shift_range = 0.2
    train_datagen.height_shift_range = 0.2
    train_datagen.shear_range = 0.2
    train_datagen.zoom_range = (0.9, 1.1)
    train_datagen.horizontal_flip = True
    train_datagen.vertical_flip = False
    train_datagen.fill_mode = 'nearest'
    # Fit the model on the augmented data
    train_generator = train_datagen.flow(train_data, train_labels, batch_size=batch_size)
    test_generator = train_datagen.flow(test_data, test_labels, batch_size=batch_size)
    # Return
    return train_generator, test_generator
# Creating the model for image classifying
def image_model():
    # Getting the required data
    train_data, train_labels, test_data, test_labels = dataset()
    train_generator, test_generator = data_generator()
    # Define the model architecture
    model = tensorflow.keras.Sequential()
    model.add(tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, num_channels)))
    model.add(tensorflow.keras.layers.MaxPooling2D((2, 2)))
    model.add(tensorflow.keras.layers.MaxPooling2D((2, 2)))
    model.add(tensorflow.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tensorflow.keras.layers.MaxPooling2D((2, 2)))
    model.add(tensorflow.keras.layers.MaxPooling2D((2, 2)))
    model.add(tensorflow.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tensorflow.keras.layers.Flatten())
    model.add(tensorflow.keras.layers.Dense(128, activation='relu'))
    model.add(tensorflow.keras.layers.Dense(1, activation='sigmoid'))
    # Compile the model
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    # Define class weights
    class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=numpy.unique(train_labels), y=train_labels)
    class_weight_dict = dict(enumerate(class_weights))
    # Train the model
    model.fit(train_generator, validation_data=test_generator, epochs=num_epochs, class_weight=class_weight_dict, verbose=0)
    # Save the model
    model.save('model/image.h5')
    # Return
    return model
# Test the model
def model_test():
    model = image_model()
    # Run the model on a test image
    for filename in os.listdir('test'):
        img = cv2.imread(os.path.join('test', filename))
        img = cv2.resize(img, (64, 64))
        img = img.astype('float32') / 255.0
        result = model.predict(numpy.array([img]), verbose=0)
        # Print the result
        if result > 0.5:
            print(f'Face detected! for file {filename}.')
        else:
            print(f'No face detected for file {filename}.')

if __name__ == "__main__":
    # Calling the model
    model_test()
