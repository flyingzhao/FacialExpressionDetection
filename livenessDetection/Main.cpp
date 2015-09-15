#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

//string face_cascade_name = "haarcascade_frontalface_alt.xml";
string face_cascade_name="D:/livenessDetection/livenessDetection/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

CascadeClassifier face_cascade;
string window_name = "faceDetection";

vector<Rect> detectAndDisplay(Mat frame);

int main(){
	Mat image;
	image = imread("2.jpg");
	resize(image,image,Size(320,320));
	if( !face_cascade.load(face_cascade_name)){ 
		printf("[error] 无法加载级联分类器文件！\n");
		return -1; 
	}
	std::vector<Rect> results;
	results=detectAndDisplay(image);

	waitKey(0);    
	}

vector<Rect> detectAndDisplay( Mat frame ){
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame,frame_gray,CV_BGR2GRAY);
	equalizeHist(frame_gray,frame_gray);

	face_cascade.detectMultiScale(frame_gray,faces,1.1,2,0|CV_HAAR_SCALE_IMAGE,Size(30,30));

	for( int i=0;i<faces.size();i++){
		Point center(faces[i].x + faces[i].width/2,faces[i].y+faces[i].height/2);
		ellipse(frame,center,Size(faces[i].width/2,faces[i].height/2),0,0,360,Scalar(255,0,255),4,8,0);
	}

	imshow(window_name,frame);
	return faces;

}
	
