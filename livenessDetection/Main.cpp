#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../lib/afreader.h"
#include "../lib/asmmodel.h"
#include "../lib/modelfile.h"
#include "../lib/modelimage.h"
#include "../lib/shapeinfo.h"
#include "../lib/shapemodel.h"
#include "../lib/shapevec.h"
#include "../lib/similaritytrans.h"

using namespace std;
using namespace cv;
using namespace StatModel;


//vector<Rect> detectAndDisplay(Mat frame);
//vector<Rect> detectFaces(Mat frame);


//detect eyes
vector<Rect> detectAndDisplay( Mat frame ){

	std::vector<Rect> eyes;
	Mat frame_gray;
	CascadeClassifier face_cascade;
	string face_cascade_name="../haarcascades/frontalEyes35x16.xml";

	if( !face_cascade.load(face_cascade_name)){ 
		printf("[error] 无法加载级联分类器文件！\n");
	}

	cvtColor(frame,frame_gray,CV_BGR2GRAY);
	equalizeHist(frame_gray,frame_gray);

	face_cascade.detectMultiScale(frame_gray,eyes,1.1,2,0|CV_HAAR_SCALE_IMAGE,Size(30,30));
	for( int i=0;i<(int)eyes.size();i++){
		
		//Point center(eyes[i].x + eyes[i].width/2,eyes[i].y+eyes[i].height/2);
		//ellipse(frame,center,Size(eyes[i].width/2,eyes[i].height/2),0,0,360,Scalar(0,255,0),4,8,0);		
		imshow("eyes",frame(eyes[0]));
		Mat eyeFrame=frame(eyes[0]);
		cvtColor(eyeFrame,eyeFrame,CV_BGR2GRAY);
		adaptiveThreshold(eyeFrame,eyeFrame,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY_INV,7,17);
		imshow("eyesThreshold",eyeFrame);
		float a=10*countNonZero(eyeFrame)/float(eyeFrame.rows*eyeFrame.cols);
		cout<<a;
		rectangle(frame,eyes[i],Scalar(0,255,0),4,8,0);
	}
	//cout<<eyes.size();
	imshow("face_detection",frame);
	return eyes;
}

int drawASM(Mat frame,vector<Rect> faces){
	imshow("ASM",frame);
	/*imshow("asmface",frame(faces[0]));*/
	ASMModel asmModel;
	//asmModel.loadFromFile("../data/color_asm68.model");
	asmModel.loadFromFile("../data/grayall_asm.model");
	vector < ASMFitResult > fitResult = asmModel.fitAll(frame, faces);
	//asmModel.showResult(frame, fitResult);
	vector< Point_<int> > V;
	for (int i=0;i<(int)fitResult.size();i++)
	{
		fitResult[i].toPointList(V);//有待改进成多个人脸
	}
	cout<<V.size();
	for (int j = 48; j < 60; j++)//0-14脸部轮廓 15-26眉毛 27-36眼睛 37-47鼻子 48-59嘴巴 共60个点
	{
		cout<<j<<" "<<V[j].x<<" "<<V[j].y<<endl;
		circle(frame,V[j],4,Scalar(0,255,0));
	}

	return 0;
}

vector<Rect> detectFaces(Mat frame){
	std::vector<Rect> faces;
	Mat frame_gray;
	CascadeClassifier face_cascade;
	string face_cascade_name="../haarcascades/haarcascade_frontalface_alt.xml";
	if( !face_cascade.load(face_cascade_name)){ 
		printf("[error] 无法加载级联分类器文件！\n");
	}

	cvtColor(frame,frame_gray,CV_BGR2GRAY);
	equalizeHist(frame_gray,frame_gray);

	face_cascade.detectMultiScale(frame_gray,faces,1.1,2,0|CV_HAAR_SCALE_IMAGE,Size(30,30));
	drawASM(frame,faces);
	for( int i=0;i<(int)faces.size();i++){	
	
		imshow("faces",frame(faces[i]));
	    rectangle(frame,faces[i],Scalar(0,255,0),4,8,0);

	}
	imshow("face",frame);
	return faces;
}

int main(){

	Mat img,image;
	img= imread("C:\\Users\\zhao\\Downloads\\jaffeimages\\jaffe\\KA.AN1.39.tiff");
	//img= imread("4.jpg");
	resize(img,image,Size(320,320));

	std::vector<Rect> eyes,faces;
	//eyes=detectAndDisplay(image);
	faces=detectFaces(image);
	//video input
	//VideoCapture capture;  
	//capture.open(0);  
	//while(true)  
	//{  
	//	Mat frame;  
	//	capture>>frame;  
	//	imshow("readvideo",frame);  
	//	std::vector<Rect> eyes;
	//	eyes=detectAndDisplay(frame);

	//	waitKey(10);  
	//}  
	//imshow("facess",img(faces[0]));

	waitKey(0);    
}