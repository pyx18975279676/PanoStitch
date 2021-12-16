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

//Affine-SIFT������ȡ�ӿ���
class CASIFT_Detector
{
public:
	CASIFT_Detector();
	~CASIFT_Detector();

private:
	siftPar siftparameters;//SIFT���ò���
	std::vector<std::vector<keypointslist>> keypoints1;//���õ���������
	std::vector<std::vector<keypointslist>> keypoints2;//���õ���������
	int num_of_tilts1;	//����

public:
	void Detect(cv::Mat& img, std::vector<cv::KeyPoint>& keypoint, cv::Mat& descriptor);//���ӿ�1
	void DetectAndMatch(cv::Mat& img1, cv::Mat& img2, std::vector<cv::KeyPoint>& keypoint1, std::vector<cv::KeyPoint>& keypoint2,
		cv::Mat& descriptor1, cv::Mat& descriptor2, std::vector<cv::DMatch> &matches);//���ӿ�2
};

