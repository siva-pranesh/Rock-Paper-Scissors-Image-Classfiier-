
from tensorflow import keras
from tensorflow.keras.preprocessing import image

saved_model_path = 'C:/Users/91909/Documents/My_Projects/Machine_Learning_&_AI/deep_learning/Rock_Paper_Scissors/Model/rock_paper_scissors_96.h5'
model = keras.models.load_model(saved_model_path)

image_path = 'C:/Users/91909/Desktop/test2.jpg'

# Read the image
test_image = image.load_img(image_path,
                            target_size=(64,64))

# Image to array
test_image = image.img_to_array(test_image)
test_image = test_image.reshape(1,64,64,3)
result = model.predict(test_image)
result.argmax()
