#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

#include "demo_lib_sift.h"
#include "library.h"
#include "frot.h"
#include "fproj.h"
#include "compute_asift_keypoints.h"
#include "compute_asift_matches.h"


using std::endl;
using std::cout;

//Affine-SIFT特征提取接口类
class CASIFT_Detector
{
public:
	CASIFT_Detector();
	~CASIFT_Detector();

private:
	siftPar siftparameters;//SIFT配置参数
	std::vector<std::vector<keypointslist>> keypoints1;//检测得到的特征点
	std::vector<std::vector<keypointslist>> keypoints2;//检测得到的特征点
	int num_of_tilts1;	//级数

public:
	void Detect(cv::Mat& img, std::vector<cv::KeyPoint>& keypoint, cv::Mat& descriptor);//检测接口1
	void DetectAndMatch(cv::Mat& img1, cv::Mat& img2, std::vector<cv::KeyPoint>& keypoint1, std::vector<cv::KeyPoint>& keypoint2,
		cv::Mat& descriptor1, cv::Mat& descriptor2, std::vector<cv::DMatch> &matches);//检测接口2
};

