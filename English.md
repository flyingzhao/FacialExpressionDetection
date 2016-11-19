#Expression recognition

Facial expression recognition with ASM and Gabor

##Platform

visual studio 2012

##Libraries

* OpenCV 2.4.10
* asmlib

##Experiment result

Flowchart:

![image](https://github.com/flyingzhao/livenessDetection/blob/master/demo/diagram.png)<br>

ASM (60 landmark points):

![image](https://github.com/flyingzhao/livenessDetection/blob/master/demo/ASMModel.PNG)<br>

Gabor feature extraction(5 scales 8 orientations):

![image](https://github.com/flyingzhao/livenessDetection/blob/master/demo/6.PNG)<br>

Feature vector(2400 dimensions):

![image](https://github.com/flyingzhao/livenessDetection/blob/master/demo/feature.PNG)<br>

Traing SVM classifier:
![image](https://github.com/flyingzhao/livenessDetection/blob/master/demo/svm.PNG)<br>
