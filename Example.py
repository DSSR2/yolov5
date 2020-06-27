import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from detect import detect
import trainer
from trainer import object_detector

# Build a new detector
object_detection = object_detector(model="Y0", classes=["Defect"])
object_detection.load_data(path="../Dataset/")
object_detection.train(epochs=5)

# Load an existing detector to retrain
object_detection = object_detector(img_size=[640, 640], multi_scale=False, data="My_data.yaml")
object_detection.train(epochs=10)

# Prediction
detect(source="./Inference/images/", weights="weights/best.pt", out="./Inference/output/")
