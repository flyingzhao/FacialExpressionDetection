#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//ASM
#include "../lib/afreader.h"
#include "../lib/asmmodel.h"
#include "../lib/modelfile.h"
#include "../lib/modelimage.h"
#include "../lib/shapeinfo.h"
#include "../lib/shapemodel.h"
#include "../lib/shapevec.h"
#include "../lib/similaritytrans.h"
//Gabor
#include "../gabor/GaborFR.h"

using namespace std;
using namespace cv;
using namespace StatModel;

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

int getGabor(vector<Point_<int>> V){
	Mat I=imread("..\\jaffe\\angry\\KA.AN1.39.tiff",0);
	normalize(I,I,1,0,CV_MINMAX,CV_32F);
	Mat showM,showMM;Mat M,MatTemp1,MatTemp2;
	Mat line;
	int iSize=50;//如果数值比较大，比如50则接近论文中所述的情况了！估计大小和处理的源图像一样！
	for(int i=0;i<1;i++)
	{
		showM.release();
		for(int j=0;j<1;j++)
		{
			Mat M1= GaborFR::getRealGaborKernel(Size(iSize,iSize),2*CV_PI,i*CV_PI/8+CV_PI/2, j,1);
			Mat M2 = GaborFR::getImagGaborKernel(Size(iSize,iSize),2*CV_PI,i*CV_PI/8+CV_PI/2, j,1);
			//加了CV_PI/2才和大部分文献的图形一样，不知道为什么！
			Mat outR,outI;
			GaborFR::getFilterRealImagPart(I,M1,M2,outR,outI);
			//	M=GaborFR::getPhase(M1,M2);
			//			M=GaborFR::getMagnitude(M1,M2);
			//			M=GaborFR::getPhase(outR,outI);
			//			M=GaborFR::getMagnitude(outR,outI);
			M=GaborFR::getMagnitude(outR,outI);
			// 			MatTemp2=GaborFR::getPhase(outR,outI);
			// 			M=outR;
			//	 M=M1;
			//resize(M,M,Size(100,100));
			normalize(M,M,0,255,CV_MINMAX,CV_8U);
			
			imshow("gabor",M);
	        
			//showM.push_back(M);
			//line=Mat::ones(4,M.cols,M.type())*255;
			//showM.push_back(line);
			for (int j = 0; j < 60; j++)//0-14脸部轮廓 15-26眉毛 27-36眼睛 37-47鼻子 48-59嘴巴 共60个点
			{
				//cout<<V[j].x<<V[j].y;
				cout<<M.at<uchar>(V[j].x,V[j].y)<<endl;
			}

		}
	/*	showM=showM.t();
		line=Mat::ones(4,showM.cols,showM.type())*255;
		showMM.push_back(showM);
		showMM.push_back(line);*/
	}
	//showMM=showMM.t();
	//imshow("saveMM",showMM);
	return 0;
}

int drawASM(Mat frame,vector<Rect> faces){
	imshow("ASM",frame);
	/*imshow("asmface",frame(faces[0]));*/
	ASMModel asmModel;
	//asmModel.loadFromFile("../data/color_asm68.model");
	asmModel.loadFromFile("../data/grayall_asm.model");
	
	Mat newframe=frame;


	vector < ASMFitResult > fitResult = asmModel.fitAll(frame, faces);
	//asmModel.showResult(frame, fitResult);
	vector< Point_<int> > V;
	for (int i=0;i<(int)fitResult.size();i++)
	{
		fitResult[i].toPointList(V);//有待改进成多个人脸
	}
	//cout<<V.size();
	for (int j = 0; j < 60; j++)//0-14脸部轮廓 15-26眉毛 27-36眼睛 37-47鼻子 48-59嘴巴 共60个点
	{
		//cout<<j<<" "<<V[j].x<<" "<<V[j].y<<endl;
		circle(frame,V[j],4,Scalar(0,255,0));
	}
    getGabor(V);//get gabor feature
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
	img= imread("..\\jaffe\\angry\\KA.AN1.39.tiff");
	//resize(img,image,Size(320,320));

	std::vector<Rect> eyes,faces;
	//eyes=detectAndDisplay(image);
	faces=detectFaces(img);
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