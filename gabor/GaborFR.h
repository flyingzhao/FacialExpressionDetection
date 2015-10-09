#pragma once
#include "opencv2/opencv.hpp"
#include <vector>
using namespace std;
using namespace cv;
class GaborFR
{
public:
	GaborFR();
	static Mat	getImagGaborKernel(Size ksize, double sigma, double theta, 
									double nu,double gamma=1, int ktype= CV_32F);
	static Mat	getRealGaborKernel( Size ksize, double sigma, double theta, 
									double nu,double gamma=1, int ktype= CV_32F);
	static Mat	getPhase(Mat &real,Mat &imag);
	static Mat	getMagnitude(Mat &real,Mat &imag);
	static void getFilterRealImagPart(Mat& src,Mat& real,Mat& imag,Mat &outReal,Mat &outImag);
	static Mat	getFilterRealPart(Mat& src,Mat& real);
	static Mat	getFilterImagPart(Mat& src,Mat& imag);

	void		Init(Size ksize=Size(19,19), double sigma=2*CV_PI,
					double gamma=1, int ktype=CV_32FC1);
private:
	vector<Mat> gaborRealKernels;
	vector<Mat> gaborImagKernels;
	bool isInited;
};