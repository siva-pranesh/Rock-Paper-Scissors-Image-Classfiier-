
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt 

saved_model_path = 'C:/Users/91909/Documents/My_Projects/Machine_Learning_&_AI/deep_learning/Rock_Paper_Scissors/Model/rock_paper_scissors_96.h5'
model = keras.models.load_model(saved_model_path)

# Read the image
image_path = 'C:/Users/91909/Desktop/test9.jfif'
input_image = image.load_img(image_path,
                            target_size=(64,64))
# Displaying the image
plt.imshow(input_image)
plt.show()

# Converting Image to array
input_image = image.img_to_array(input_image)
input_image = input_image.reshape(1,64,64,3)
result = model.predict(input_image)

# training_set.class_indices
class_dict = {'paper': 0, 'rock': 1, 'scissors': 2}

# Returning the key based on value input
key_list = list(class_dict.keys())
val_list = list(class_dict.values())
position = val_list.index(result.argmax())
print('^'*25)
print('This is', key_list[position])
print('^'*25)