# # from flask import Flask, request, jsonify
# # import tensorflow as tf
# # import numpy as np
# # import requests
# # from io import BytesIO
# # from PIL import Image
# # import cv2
# # import os

# # app = Flask(__name__)

# # # Configuration dictionary with image size and class names
# # CONFIGURATION = {
# #     "IM_SIZE": 256,
# #     "CLASS_NAMES": ["ACCESSORIES", "BRACELETS", "CHAIN", "CHARMS", "EARRINGS",
# #                     "ENGAGEMENT RINGS", "ENGAGEMENT SET", "FASHION RINGS", "NECKLACES", "WEDDING BANDS"],
# # }

# # # Global model and inference function
# # model = None

# # def init_model():
# #     global model
# #     model_path = os.path.join('models', 'update_lenet_model_save.h5')
# #     # Load the Keras model
# #     model = tf.keras.models.load_model(model_path)
# #     print("Model loaded successfully.")

# # def preprocess_image(image):
# #     # Resize and preprocess the image for the model
# #     image = cv2.resize(image, (CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]))
# #     image = tf.convert_to_tensor(image, dtype=tf.float32)  # Ensure TensorFlow tensor
# #     image = image / 255.0  # Normalize the image
# #     image = tf.expand_dims(image, axis=0)  # Add batch dimension
# #     return image

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     try:
# #         # Parse the input data
# #         data = request.get_json()
        
# #         if 'image_url' in data:
# #             # Load image from the URL
# #             response = requests.get(data['image_url'])
# #             image = np.array(Image.open(BytesIO(response.content)).convert('RGB'))
# #         elif 'image_data' in data:
# #             # Load image from the provided image data
# #             image = np.array(data['image_data'], dtype=np.uint8)
# #         else:
# #             return jsonify({"error": "No valid input image provided."}), 400
        
# #         # Preprocess the image
# #         image = preprocess_image(image)
        
# #         # Make predictions
# #         predictions = model.predict(image)
        
# #         # Get the top 3 predicted classes and their probabilities
# #         predictions = predictions[0]  # Remove batch dimension
# #         top_3_indices = np.argsort(predictions)[-3:][::-1]  # Top 3 indices
# #         top_3_probabilities = predictions[top_3_indices]
# #         top_3_classes = [CONFIGURATION['CLASS_NAMES'][index] for index in top_3_indices]
        
# #         # Prepare the results as a list of dictionaries
# #         top_3_predictions = [
# #             {"class_name": top_3_classes[i], "probability": float(top_3_probabilities[i])}
# #             for i in range(3)
# #         ]
        
# #         # Return the JSON response
# #         return jsonify({"top_3_classes_predictions": top_3_predictions})
    
# #     except Exception as e:
# #         return jsonify({"error": str(e)}), 500

# # # Add the GET route to say "Hello"
# # @app.route('/', methods=['GET'])
# # def hello():
# #     return "Hello! Welcome to product type classification API with Keras."

# # if __name__ == '__main__':
# #     init_model()  # Initialize the model before starting the server
# #     app.run()  # Defaults to host='127.0.0.1', port=5000


# #----------------------------------------------------------------------------------------------------
# #-----------------------------------------------------------------------------------------------------
# # from flask import Flask, request, jsonify
# # import tensorflow as tf
# # import numpy as np
# # import requests
# # from io import BytesIO
# # from PIL import Image
# # import cv2
# # import os
# # from dotenv import load_dotenv

# # # Load environment variables from .env file
# # load_dotenv()

# # app = Flask(__name__)

# # # Configuration dictionary with image size and class names
# # CONFIGURATION = {
# #     "IM_SIZE": 256,
# #     "CLASS_NAMES": ["ACCESSORIES", "BRACELETS", "CHAIN", "CHARMS", "EARRINGS",
# #                     "ENGAGEMENT RINGS", "ENGAGEMENT SET", "FASHION RINGS", "NECKLACES", "WEDDING BANDS"],
# # }

# # # Local model path
# # LOCAL_MODEL_PATH = 'models/update_lenet_model_save.h5'

# # # Global model and inference function
# # model = None

# # def init_model():
# #     global model
# #     # Check if the model file exists
# #     if not os.path.exists(LOCAL_MODEL_PATH):
# #         raise FileNotFoundError(f"Model file not found at {LOCAL_MODEL_PATH}")
# #     # Load the Keras model
# #     model = tf.keras.models.load_model(LOCAL_MODEL_PATH)
# #     print("Model loaded successfully.")

# # def preprocess_image(image):
# #     # Resize and preprocess the image for the model
# #     image = cv2.resize(image, (CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]))
# #     image = tf.convert_to_tensor(image, dtype=tf.float32)  # Ensure TensorFlow tensor
# #     image = image / 255.0  # Normalize the image
# #     image = tf.expand_dims(image, axis=0)  # Add batch dimension
# #     return image

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     try:
# #         # Parse the input data
# #         data = request.get_json()
        
# #         if 'image_url' in data:
# #             # Load image from the URL
# #             response = requests.get(data['image_url'])
# #             image = np.array(Image.open(BytesIO(response.content)).convert('RGB'))
# #         elif 'image_data' in data:
# #             # Load image from the provided image data
# #             image = np.array(data['image_data'], dtype=np.uint8)
# #         else:
# #             return jsonify({"error": "No valid input image provided."}), 400
        
# #         # Preprocess the image
# #         image = preprocess_image(image)
        
# #         # Make predictions
# #         predictions = model.predict(image)
        
# #         # Get the top 3 predicted classes and their probabilities
# #         predictions = predictions[0]  # Remove batch dimension
# #         top_3_indices = np.argsort(predictions)[-3:][::-1]  # Top 3 indices
# #         top_3_probabilities = predictions[top_3_indices]
# #         top_3_classes = [CONFIGURATION['CLASS_NAMES'][index] for index in top_3_indices]
        
# #         # Prepare the results as a list of dictionaries
# #         top_3_classes_predictions = [
# #             {"class_name": top_3_classes[i], "probability": float(top_3_probabilities[i])}
# #             for i in range(3)
# #         ]
        
# #         # Return the JSON response
# #         return jsonify({"top_3_classes_predictions": top_3_classes_predictions})
    
# #     except Exception as e:
# #         return jsonify({"error": str(e)}), 500

# # @app.route('/', methods=['GET'])
# # def hello():
# #     return "Hello! Welcome to product type classification API with Keras."

# # if __name__ == '__main__':
# #     init_model()  # Initialize the model before starting the server
# #     app.run(debug=False)



# #__________________________________________________________________________________________________-

# #_______________________________________________________________________________________________________________-
# ###########################USING SAS TOKEN TO ACCESS THE MODEL ######################################################
# from flask import Flask, request, jsonify
# import tensorflow as tf
# import numpy as np
# import requests
# from io import BytesIO
# from PIL import Image
# import cv2
# import os
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# app = Flask(__name__)

# # Configuration dictionary with image size and class names
# CONFIGURATION = {
#     "IM_SIZE": 256,
#     "CLASS_NAMES": ["ACCESSORIES", "BRACELETS", "CHAIN", "CHARMS", "EARRINGS",
#                     "ENGAGEMENT RINGS", "ENGAGEMENT SET", "FASHION RINGS", "NECKLACES", "WEDDING BANDS"],
# }

# # Fetch environment variable for Azure Blob Storage SAS URL
# BLOB_SAS_URL = os.getenv("BLOB_SAS_URL")

# # Global model and inference function
# model = None

# def init_model():
#     global model
#     # Load the Keras model directly from Azure Blob Storage using the SAS URL
#     model = tf.keras.models.load_model(BLOB_SAS_URL)
#     print("Model loaded successfully from Azure Blob Storage.")

# def preprocess_image(image):
#     # Resize and preprocess the image for the model
#     image = cv2.resize(image, (CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]))
#     image = tf.convert_to_tensor(image, dtype=tf.float32)  # Ensure TensorFlow tensor
#     image = image / 255.0  # Normalize the image
#     image = tf.expand_dims(image, axis=0)  # Add batch dimension
#     return image

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Parse the input data
#         data = request.get_json()
        
#         if 'image_url' in data:
#             # Load image from the URL
#             response = requests.get(data['image_url'])
#             image = np.array(Image.open(BytesIO(response.content)).convert('RGB'))
#         elif 'image_data' in data:
#             # Load image from the provided image data
#             image = np.array(data['image_data'], dtype=np.uint8)
#         else:
#             return jsonify({"error": "No valid input image provided."}), 400
        
#         # Preprocess the image
#         image = preprocess_image(image)
        
#         # Make predictions
#         predictions = model.predict(image)
        
#         # Get the top 3 predicted classes and their probabilities
#         predictions = predictions[0]  # Remove batch dimension
#         top_3_indices = np.argsort(predictions)[-3:][::-1]  # Top 3 indices
#         top_3_probabilities = predictions[top_3_indices]
#         top_3_classes = [CONFIGURATION['CLASS_NAMES'][index] for index in top_3_indices]
        
#         # Prepare the results as a list of dictionaries
#         top_3_classes_predictions = [
#             {"class_name": top_3_classes[i], "probability": float(top_3_probabilities[i])}
#             for i in range(3)
#         ]
        
#         # Return the JSON response
#         return jsonify({"top_3_classes_predictions": top_3_classes_predictions})
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/', methods=['GET'])
# def hello():
#     return "Hello! Welcome to product type classification API with Keras."

# if __name__ == '__main__':
#     init_model()  # Initialize the model before starting the server
#     app.run(debug=False)

#################################################################################################3
from flask import Flask, request, jsonify
import tensorflow as tf
import requests
from io import BytesIO
import numpy as np
from PIL import Image
import tempfile
import os

app = Flask(__name__)

# Set the environment variable for the Blob SAS URL
BLOB_SAS_URL = os.getenv('BLOB_SAS_URL')

# Global variable to hold the model
model = None

# Function to preprocess the image
def preprocess_image(image):
    # Adjust the preprocessing as required by your model
    image = tf.image.resize(image, (224, 224))  # Example resize, adjust as necessary
    image = image / 255.0  # Normalize to [0, 1] range
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to load the model
def init_model():
    global model
    try:
        # Download the model file from Azure Blob Storage
        response = requests.get(BLOB_SAS_URL)
        if response.status_code == 200:
            model_content = BytesIO(response.content)

            # Use a temporary file to save the model content
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as temp_file:
                temp_file.write(model_content.read())
                temp_file_path = temp_file.name

            # Load the model from the temporary file
            model = tf.keras.models.load_model(temp_file_path)
            print("Model loaded successfully from Azure Blob Storage.")
            os.remove(temp_file_path)  # Clean up temporary file
        else:
            print(f"Failed to download model from Azure Blob Storage. Status code: {response.status_code}")

    except Exception as e:
        print(f"Error loading model: {str(e)}")

# Initialize the model when the application starts
init_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model is not loaded."}), 500

        # Parse the input data
        data = request.get_json()

        if 'image_url' in data:
            # Load image from the URL
            response = requests.get(data['image_url'])
            image = np.array(Image.open(BytesIO(response.content)).convert('RGB'))
        elif 'image_data' in data:
            # Load image from the provided image data
            image = np.array(data['image_data'], dtype=np.uint8)
        else:
            return jsonify({"error": "No valid input image provided."}), 400

        # Preprocess the image
        image = preprocess_image(image)

        # Make predictions
        predictions = model.predict(image)

        # Get the top 3 predicted classes and their probabilities
        predictions = predictions[0]  # Remove batch dimension
        top_3_indices = np.argsort(predictions)[-3:][::-1]  # Top 3 indices
        top_3_probabilities = predictions[top_3_indices]
        # Replace CONFIGURATION['CLASS_NAMES'] with your actual class names
        top_3_classes = [CONFIGURATION['CLASS_NAMES'][index] for index in top_3_indices]

        # Prepare the results as a list of dictionaries
        top_3_classes_predictions = [
            {"class_name": top_3_classes[i], "probability": float(top_3_probabilities[i])}
            for i in range(3)
        ]

        # Return the JSON response
        return jsonify({"top_3_classes_predictions": top_3_classes_predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
