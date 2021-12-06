import cv2
import numpy as np


path_weights = 'weights/frozen_v_124_2021-09-22.pb'
path_classes = 'weights/classes.txt'

class Driver(object):
    def __init__(self, path_weights, path_classes):
        self.net = cv2.dnn.readNet(path_weights)
        self.layer = 'StatefulPartitionedCall/StatefulPartitionedCall/model/cls/Softmax'
        self.class_names = self.read_classes(path_classes)
    
    def read_classes(self, path):             
        with open(f'{path}', 'r') as f:
            class_labels = f.readlines()
        class_labels = [cls.strip() for cls in class_labels]
        return class_labels    
    
    def predictions_classes(self, input_image:str):
        image = cv2.imread(input_image, cv2.IMREAD_COLOR)
        img_blob = cv2.dnn.blobFromImage(image, 1/255., (224,224), swapRB=True, crop=False)
        self.net.setInput(img_blob)
        output = self.net.forward(self.layer)
        pred_cls = np.argmax(output)
        classNames = self.class_names[pred_cls]
        return print(classNames.capitalize())



cls = Driver(path_weights,path_classes)


