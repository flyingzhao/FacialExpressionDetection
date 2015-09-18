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

//string face_cascade_name = "../haarcascades/haarcascade_frontalface_alt.xml";
//haarcascade_eye
//haarcascade_eye_tree_eyeglasses
//haarcascade_lefteye_2splits
//haarcascade_mcs_eyepair_small
//haarcascade_mcs_eyepair_big
//haarcascade_mcs_lefteye
//haarcascade_mcs_righteye
//haarcascade_righteye_2splits  perform good
//haarcascade_lefteye_2splits
//string face_cascade_name="../haarcascades/haarcascade_eye.xml";

//frontalEyes35x16
//
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
	imshow("asmface",frame(faces[0]));
	ASMModel asmModel;
	asmModel.loadFromFile("../data/color_asm68.model");
	vector < ASMFitResult > fitResult = asmModel.fitAll(frame, faces);
	asmModel.showResult(frame, fitResult);
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
	img= imread("4.jpg");
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
	StatModel::ASMModel asmModel;

	waitKey(0);    
}