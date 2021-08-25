
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------------
training_path = 'C:/Users/91909/Documents/My_Projects/Machine_Learning_&_AI/deep_learning/Rock_Paper_Scissors/Train_Test_Validation/train'
testing_path = 'C:/Users/91909/Documents/My_Projects/Machine_Learning_&_AI/deep_learning/Rock_Paper_Scissors/Train_Test_Validation/test'
validation_path = 'C:/Users/91909/Documents/My_Projects/Machine_Learning_&_AI/deep_learning/Rock_Paper_Scissors/Train_Test_Validation/val'
#-------------------------------------------------------------------------------------------
training_datagen = ImageDataGenerator(rescale = 1/255, rotation_range=0.2,
                                   shear_range=0.3,zoom_range=0.2,
                                   vertical_flip=True,horizontal_flip=True)

testing_datagen = ImageDataGenerator(rescale = 1/255)
validation_datagen = ImageDataGenerator(rescale = 1/255)
#-------------------------------------------------------------------------------------------
# Creating pipeline
training_set = training_datagen.flow_from_directory(training_path,
                                                 target_size=(64,64),
                                                 batch_size=32,class_mode='categorical')
testing_set = training_datagen.flow_from_directory(testing_path,
                                                 target_size=(64,64),
                                                 batch_size=32,class_mode='categorical')
validation_set = training_datagen.flow_from_directory(validation_path,
                                                 target_size=(64,64),
                                                 batch_size=32,class_mode='categorical')
#-------------------------------------------------------------------------------------------
# Reviewing sample data
training_data = next(training_set)
training_data[0].shape
plt.imshow(training_data[0][0])
plt.show()
#-------------------------------------------------------------------------------------------
# Building the CNN Model
classifier = Sequential()

classifier.add(Conv2D(filters=30,kernel_size=3,activation='relu',padding='valid',input_shape=(64, 64, 3)))
classifier.add(MaxPooling2D())
classifier.add(Conv2D(filters=60,kernel_size=3,activation='relu',padding='valid'))
classifier.add(MaxPooling2D())
classifier.add(Conv2D(filters=120,kernel_size=3,activation='relu',padding='valid'))
classifier.add(MaxPooling2D())

classifier.add(Flatten())

classifier.add(Dense(224,activation='relu'))
tensorflow.keras.layers.Dropout(.2)
classifier.add(Dense(224,activation='sigmoid'))
tensorflow.keras.layers.Dropout(.2)
classifier.add(Dense(224,activation='relu'))
tensorflow.keras.layers.Dropout(.2)
classifier.add(Dense(3,activation='softmax'))

classifier.compile(loss = 'categorical_crossentropy',
    optimizer = tensorflow.optimizers.Adam(learning_rate=0.001),
    metrics = ['accuracy'])
#-------------------------------------------------------------------------------------------
training_change = classifier.fit(training_set, epochs=10, validation_data = validation_set)
print(training_change.history.keys())

#-------------------------------------------------------------------------------------------
# Plotting the Training/Validation data loss and accuracy
plt.title('Training vs validation loss')
plt.plot(training_change.history['val_loss'],label='val_loss')
plt.plot(training_change.history['loss'],label='loss')
plt.legend()
plt.show()

plt.title('Training vs validation accuracy')
plt.plot(training_change.history['accuracy'],label='Accuracy')
plt.plot(training_change.history['val_accuracy'],label='Val_Accuracy')
plt.legend()
plt.show()
#-------------------------------------------------------------------------------------------
classifier.summary()
#-------------------------------------------------------------------------------------------
classifier.evaluate(testing_set)
# Model1_Accuracy: 0.95

classifier.predict(testing_set)
#-------------------------------------------------------------------------------------------
## Hyperparameter tuning
#-------------------------------------------------------------------------------------------

def model_builder(hp):
    classifier = Sequential()
    classifier.add(Conv2D(filters=30,kernel_size=3,activation='relu',padding='valid'))
    classifier.add(MaxPooling2D())
    classifier.add(Conv2D(filters=60,kernel_size=3,activation='relu',padding='valid'))
    classifier.add(MaxPooling2D())
    classifier.add(Conv2D(filters=120,kernel_size=3,activation='relu',padding='valid'))
    classifier.add(MaxPooling2D())
    classifier.add(Flatten())
    # Tune the number of units in the Dense layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    classifier.add(Dense(units=hp_units,activation='relu'))
    tensorflow.keras.layers.Dropout(.2)
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    classifier.add(Dense(units=hp_units,activation='sigmoid'))
    tensorflow.keras.layers.Dropout(.2)
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    classifier.add(Dense(units=hp_units,activation='relu'))
    tensorflow.keras.layers.Dropout(.2)
    classifier.add(Dense(3,activation='softmax'))
    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    classifier.compile(loss = 'categorical_crossentropy',
    optimizer = tensorflow.optimizers.Adam(learning_rate=hp_learning_rate),
    metrics = ['accuracy'])
    return classifier

from keras_tuner.tuners import RandomSearch
import os

tuner = RandomSearch(
    model_builder,
    objective='val_accuracy',
    max_trials=32,
    overwrite=True,
    directory=os.path.normpath('C:/')
)

tuner.search(training_set,
             validation_data=(validation_set),
             epochs=10, batch_size=32)

#-------------------------------------------------------------------------------------------
best_model = tuner.get_best_models()[0]
best_model.evaluate(training_set)
# Best Trial summary
# Hyperparameters:
# units: 224
# learning_rate: 0.001
# Score: 0.9839816689491272
# Best_Model_Training_Accuracy: 0.97
# Best_Model_Validation_Accuracy: 0.98
# Best_Model_Testing_Accuracy: 0.96
#-------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------
## Saving the best model
#-------------------------------------------------------------------------------------------

# create a HDF5 file 'rock_paper_scissors.h5'
classifier.save('rock_paper_scissors_96.h5')

# training_set.class_indices
# {'paper': 0, 'rock': 1, 'scissors': 2}