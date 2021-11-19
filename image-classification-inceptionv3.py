# Libraries
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle


# Parameters
epochs = 20
batch_size = 32
img_size = 224
num_classes = 177

train_dir = "data/images_224x224/train"
validation_dir = "data/images_224x224/test"


# Necessary functions
def print_imgs(directory):
    """
    Show 6 example from the dataset
    """
    plt.figure(figsize=(10, 10))
    folder = directory + "/" + os.listdir(directory)[58]
    for i, cat_id in enumerate(os.listdir(folder)):
        img_dir = folder + "/" + cat_id
        plt.subplot(2, 3, i+1)
        plt.imshow(imread(img_dir))
        plt.axis('off')
        if i == 5:
            break
    plt.tight_layout()
    plt.show()


def plot_hist(hist):
    """
    Plot the training history
    """
    plt.plot(hist.history["acc"])
    plt.plot(hist.history["val_acc"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


def build_model(num_classes):
    """
    Build the InceptionV3 model
    """
    inception_model = InceptionV3(include_top=False, input_shape=(img_size, img_size, 3))
    for layer in inception_model.layers:
        layer.trainable=False
    flat1 = Flatten()(inception_model.layers[-1].output)
    class1 = Dense(1024, activation='relu')(flat1)
    dropout = Dropout(0.2)(class1)
    output = Dense(num_classes, activation='softmax')(dropout)
    model = Model(inputs = inception_model.inputs, outputs = output)
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=["acc"])
    return model


def save_labels_dict(obj, name):
    """
    Save a dictionary file
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# Print example images from train dataset
print_imgs(train_dir)


# Data generators
# For the train set
train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)


# For the validation set
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    target_size=(img_size, img_size))
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        target_size=(img_size, img_size))


# Model
model = build_model(num_classes=num_classes)
hist = model.fit(train_generator, epochs=epochs, validation_data=validation_generator, verbose=1)
plot_hist(hist)


# Test the model on validation set
valid_loss, valid_accuracy = model.evaluate(validation_generator, verbose=1)
print('Validation Accuracy: ', round((valid_accuracy * 100), 2), "%")


# Save the model
model.save('inceptionV3_model.h5')


# Save the dictionary that hold category ids
labels_dict = train_generator.class_indices,  
save_labels_dict(labels_dict, "labels")