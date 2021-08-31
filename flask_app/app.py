from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

dic = { 0:'Paper', 1:'Rock', 2:'Scissor'}

saved_model_path = 'C:/Users/91909/Documents/My_Projects/Machine_Learning_&_AI/deep_learning/Rock_Paper_Scissors/Model/save_model.h5'
model = load_model(saved_model_path)

model.make_predict_function()

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(64,64))
	i = image.img_to_array(i)
	i = i.reshape(1,64,64,3)
	result = model.predict(i)
	p = np.argmax(result,axis=1)
	return dic[p[0]]

@app.route("/rock-paper-scissors-image-classifier-model-demo-siva-pranesh", methods=['GET','POST'])
def main():
    return render_template("local_site.html")

@app.route("/rock-paper-scissors-image-classifier-model-demo-siva-pranesh/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "static/" + img.filename	
		result = predict_label(img_path)
	return render_template("local_site.html", prediction = result, img_path = img_path)


if __name__ =='__main__':
		app.run(debug = True)