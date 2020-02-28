# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import flask 

# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(200, 200))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 200, 200, 3)
	# center pixel data
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img

# load an image and predict the class
def run_example():
	# load the image
    img = load_image('sample_image.jpg')
	# load model
    model = load_model('model.h5')
	# predict the class
    result = model.predict(img)
    print(result[0])
    return result

# entry point, run the example
run_example()


# Every ML/DL model has a specific format 
# of taking input. Before we can predict on 
# the input image, we first need to preprocess it. 
def prepare_image(image, target): 
    if image.mode != "RGB": 
        image = image.convert("RGB") 
	
	# Resize the image to the target dimensions 
    image = image.resize(target) 
	
	# PIL Image to Numpy array 
    image = img_to_array(image) 
	
	# Expand the shape of an array, 
	# as required by the Model 
    image = np.expand_dims(image, axis = 0) 
	
	# preprocess_input function is meant to 
	# adequate your image to the format the model requires 
    image = imagenet_utils.preprocess_input(image) 

	# return the processed image 
    return image 



# Now, we can predict the results. 
@app.route("/predict", methods =["POST"]) 
def predict(): 
    data = {} # dictionary to store result 
    data["success"] = False

	# Check if image was properly sent to our endpoint 
    if flask.request.method == "POST": 
        if flask.request.files.get("image"): 
            image = flask.request.files["image"].read() 
            image = Image.open(io.BytesIO(image)) 

			# Resize it to 200x200 pixels 
			# (required input dimensions for ResNet) 
            image = prepare_image(image, target =(200, 200)) 

		# Predict ! global preds, results 
        # load model
        model = load_model('model.h5')
        preds = model.predict(image) 
        results = imagenet_utils.decode_predictions(preds) 
        data["predictions"] = [] 

		
        for (ID, label, probability) in results[0]: 
            r = {"label": label, "probability": float(probability)} 
            data["predictions"].append(r) 

            data["success"] = True

	# return JSON response 
    return flask.jsonify(data) 
