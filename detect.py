from imageai.Detection import ObjectDetection
import os
import time

dir = ObjectDetection()
dir.setModelTypeAsRetinaNet()
dir.setModelPath(os.path.abspath('retinanet.h5'))
dir.loadModel(detection_speed='faster')

objects = dir.CustomObjects(
    car=True, bus=True, train=True, truck=True, traffic_light=True, stop_sign=True)

print('----- detecting -----')
t = time.time()

det = dir.detectCustomObjectsFromImage(
    custom_objects=objects,
    input_image=os.path.abspath('imageai_test_2.jpg'),
    output_image_path=os.path.abspath('imageai_test_2_res.jpg'))

print(time.time() - t)

for each_obj in det:
    print(each_obj)
