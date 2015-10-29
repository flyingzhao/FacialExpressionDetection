#表情识别
<strong>基于ASM和Gabor的人脸表情识别</strong><br>
平台: Windows 7<br>
集成开发环境: Visual Studio 2012<br>
库: 
<ul>
<li>OpenCV2.4.10</li>
<li>asmlib</li>
</ul>
工具：
<ul>
<li>CMake</li>
<li>VisualAssistX</li>
</ul>
流程图<br/>
![image](https://github.com/flyingzhao/livenessDetection/blob/master/demo/diagram.png)<br>
ASM特征点标记(提取60个特征点)<br>
![image](https://github.com/flyingzhao/livenessDetection/blob/master/demo/ASMModel.PNG)<br>
Gabor特征提取（5个尺度，8个方向）<br/>
![image](https://github.com/flyingzhao/livenessDetection/blob/master/demo/6.PNG)<br>
特征向量生成（2400维）<br/>
![image](https://github.com/flyingzhao/livenessDetection/blob/master/demo/feature.PNG)<br>
分类器训练（使用SVM）<br/>
![image](https://github.com/flyingzhao/livenessDetection/blob/master/demo/svm.PNG)<br>
