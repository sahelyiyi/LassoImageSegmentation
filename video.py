from imageai.Detection import ObjectDetection
import os
import cv2
import numpy as np
from lasso import *
from lasso import  _get_image_graph, _get_predicted_img
import imageio


detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join('/Users/sahel/Downloads', "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

for i in range(1000):
    print (i)
    img_path = os.path.join('./frames/', "frame%d.jpg" % i)

    detections = detector.detectObjectsFromImage(input_image=img_path, output_image_path=os.path.join('./', "imagenew.jpg"))

    img = cv2.imread(img_path)
    bw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    h, w, _ = img.shape
    lables = np.zeros((h, w))
    for eachObject in detections:
        # print(eachObject["name"], " : ", eachObject["percentage_probability"] , eachObject['box_points'])
        if eachObject['name'] == 'car':
            # cars.append(eachObject['box_points'])
            box_points = eachObject['box_points']
            for x in range(box_points[0] - 1, box_points[2]):
                for y in range(box_points[1] - 1, box_points[3]):
                    if x >= 854 or y >= 480:
                        continue
                    lables[y, x] = 1
    np.save('./new_labels/label%d' % i, lables)

treshhold = 170


for y in range(h):
    for x in range(w):
        if bw_img[y, x] >= treshhold:
            lables[y, x] = 1


img_labels = 2*lables-1
img_labels = img_labels.astype(int)

show_img_with_label(img)
show_img_with_label(img,img_labels)

uc_graph, uc_opt_acc, uc_y, uc_X, M = _get_image_graph(img, img_labels)
# prob = [0.01, 0.1, 0.5, 1.0]
prob = [0.1]
uc_non_acc, uc_acc = train(uc_graph, uc_X, uc_y, uc_opt_acc, 10, reg=reg, prob=prob, num_exps=1, M=None, is_plot=True, prefix="img",
                               is_saved=True, compare_with_lr=True)

predicted_img, predicted_vanila_img, lables, vanila_lables = _get_predicted_img(img)

print('missed lables by out method for reg: 0.01')
show_img_with_label(predicted_img, 'our_method')

print('missed lables for vanila logistic regression')
show_img_with_label(predicted_vanila_img, 'vanila')
