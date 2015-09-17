#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

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

vector<Rect> detectAndDisplay(Mat frame);

int main(){

	Mat img,image;
	img= imread("1.jpg");
	resize(img,image,Size(320,320));
	
	std::vector<Rect> faces;
	faces=detectAndDisplay(image);
	//video input
	//VideoCapture capture;  
	//capture.open(0);  
	//while(true)  
	//{  
	//	Mat frame;  
	//	capture>>frame;  
	//	imshow("readvideo",frame);  
	//	std::vector<Rect> faces;
	//	faces=detectAndDisplay(frame);

	//	waitKey(10);  
	//}  

	waitKey(0);    
	}

//detect faces
vector<Rect> detectAndDisplay( Mat frame ){

	std::vector<Rect> faces;
	Mat frame_gray;
	CascadeClassifier face_cascade;
	string face_cascade_name="../haarcascades/frontalEyes35x16.xml";

	if( !face_cascade.load(face_cascade_name)){ 
		printf("[error] 无法加载级联分类器文件！\n");
	}

	cvtColor(frame,frame_gray,CV_BGR2GRAY);
	equalizeHist(frame_gray,frame_gray);

	face_cascade.detectMultiScale(frame_gray,faces,1.1,2,0|CV_HAAR_SCALE_IMAGE,Size(30,30));
	for( int i=0;i<faces.size();i++){
		
		//Point center(faces[i].x + faces[i].width/2,faces[i].y+faces[i].height/2);
		//ellipse(frame,center,Size(faces[i].width/2,faces[i].height/2),0,0,360,Scalar(0,255,0),4,8,0);		
		imshow("eyes",frame(faces[0]));
		Mat eyeFrame=frame(faces[0]);
		cvtColor(eyeFrame,eyeFrame,CV_BGR2GRAY);
		adaptiveThreshold(eyeFrame,eyeFrame,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY_INV,7,17);
		imshow("eyesThreshold",eyeFrame);
		float a=10*countNonZero(eyeFrame)/float(eyeFrame.rows*eyeFrame.cols);
		cout<<a;
		rectangle(frame,Point(faces[i].x,faces[i].y),Point(faces[i].x+faces[i].width,faces[i].y+faces[i].height),Scalar(0,255,0),4,8,0);
	}

	//cout<<faces.size();
	imshow("face_detection",frame);
	
	return faces;

}
	
