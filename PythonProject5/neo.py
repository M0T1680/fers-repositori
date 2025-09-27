from PIL import Image
import tensorflow as tf
import cv2
import os

DATASET_PATH = 'dataset/'
num_classes = len(os.listdir(DATASET_PATH))
class_mode = "binary" if num_classes == 2 else "multiclass"

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} doesn't exist")
        return
    try:
        img = Image.open(image_path)
        img.verify()
        img = Image.open(image_path)
    except (OSError, IOError):
        print(f"Error, damaged file by path {image_path}")
        return

    model = tf.keras.models.load_model('image_classifier.h5')
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Couldnt read image {image_path}")
        return
    img = cv2.resize(img, (128, 128))
    img = img/255
    img = tf.expand_dims(img, axis= 0)

    prediction = model.predict(img)
    class_names = os.listdir(DATASET_PATH)
    if class_mode == "binary":
        predicted_class = class_names[int(bool(prediction[0] > 0.5))]
    else:
        predicted_class = class_names[tf.argmax(prediction, axis = 0).numpy()[0]]
    print(f"Models predicted class: {predicted_class}")

predict_image("dataset/dog/image_dog.jpeg")
