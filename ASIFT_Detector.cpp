#include "ASIFT_Detector.h"


CASIFT_Detector::CASIFT_Detector()
{
	default_sift_parameters(this->siftparameters);	//初始化SIFT运行参数
	this->num_of_tilts1 = 7;

}


CASIFT_Detector::~CASIFT_Detector()
{
}


void CASIFT_Detector::Detect(cv::Mat& img, std::vector<cv::KeyPoint>& keypoint, cv::Mat& descriptor)
{
	if (!img.data)
	{
		std::cout << "no image data!" << std::endl;
		return;
	}
	if (1 != img.channels())
	{
		std::cout << "input image must be gray scale！"  << endl;
		return;
	}
	if (CV_32FC1 != img.type())
	{
		std::cout << "input image must be float format&single channel!" << endl;
		return;
	}

	//将图像转换为一维数组
	int rows = img.rows;
	int cols = img.cols;
	float* data = nullptr;
	std::vector<float> image(rows*cols, 0);
	for (int i = 0; i < rows; ++i)
	{
		data = img.ptr<float>(i);
		for (int j = 0; j < cols; ++j)
		{
			image[i*cols + j] = *data++;
		}
	}

	int num_keys=0;	//特征点的数目
	num_keys = compute_asift_keypoints(image, rows, cols, this->num_of_tilts1, 0, this->keypoints1, this->siftparameters);
	cout << "detected " << num_keys << " featrue points" << endl;

	int feature_count=0;	//特征图中特征的计数
	cv::Mat dst(num_keys, (int)VecLength, CV_32FC1, cv::Scalar::all(0));
	data = nullptr;
	for (int tt = 0; tt < (int)this->keypoints1.size(); tt++)
	{
		for (int rr = 0; rr < (int)this->keypoints1[tt].size(); rr++)
		{
			keypointslist::iterator ptr = this->keypoints1[tt][rr].begin();
			for (int i = 0; i < (int)this->keypoints1[tt][rr].size(); i++, ptr++)
			{
				cv::KeyPoint feature_point;
				feature_point.angle = ptr->angle;
				feature_point.pt.x = ptr->x;
				feature_point.pt.y = ptr->y;
				feature_point.size = ptr->scale;
				//cout << ptr->x << "  " << ptr->y << "  " << ptr->scale << "  " << ptr->angle;

				data = dst.ptr<float>(feature_count++);
				for (int ii = 0; ii < (int)VecLength; ii++)
				{
					*data++ = ptr->vec[ii];
					//cout << "  " << ptr->vec[ii];
				}
				keypoint.push_back(feature_point);
				//cout << std::endl;
			}
		}
	}

	descriptor = dst;
}


void CASIFT_Detector::DetectAndMatch(cv::Mat& img1, cv::Mat& img2, std::vector<cv::KeyPoint>& keypoint1, std::vector<cv::KeyPoint>& keypoint2, cv::Mat& descriptor1, cv::Mat& descriptor2, std::vector<cv::DMatch> &matches)
{
	if (!img1.data || !img2.data)
	{
		std::cout << "no image data!" << std::endl;
		return;
	}
	if (1 != img1.channels() || 1 != img2.channels())
	{
		std::cout << "input image must be gray scale！" << endl;
		return;
	}
	if (CV_32FC1 != img1.type() || CV_32FC1 != img2.type())
	{
		std::cout << "input image must be float format&single channel!" << endl;
		return;
	}

	//将图像转换为一维数组
	int rows1 = img1.rows;
	int cols1 = img1.cols;
	int rows2 = img2.rows;
	int cols2 = img2.cols;
	float* data1 = nullptr;
	float* data2 = nullptr;
	std::vector<float> image1(rows1 * cols1, 0);
	std::vector<float> image2(rows2 * cols2, 0);
	for (int i = 0; i < rows1; ++i)
	{
		data1 = img1.ptr<float>(i);
		for (int j = 0; j < cols1; ++j)
		{
			image1[i * cols1 + j] = *data1++;
		}
	}
	for (int i = 0; i < rows2; ++i)
	{
		data2 = img2.ptr<float>(i);
		for (int j = 0; j < cols2; ++j)
		{
			image2[i * cols2 + j] = *data2++;
		}
	}

	int num_keys1 = 0;	//特征点的数目
	int num_keys2 = 0;	//特征点的数目
	num_keys1 = compute_asift_keypoints(image1, rows1, cols1, this->num_of_tilts1, 0, this->keypoints1, this->siftparameters);
	num_keys2 = compute_asift_keypoints(image2, rows2, cols2, this->num_of_tilts1, 0, this->keypoints2, this->siftparameters);
	cout << "detected_1 " << num_keys1 << " featrue points" << endl;
	cout << "detected_2 " << num_keys2 << " featrue points" << endl;

	int num_matches = 0;
	matchingslist matchings;
	num_matches = compute_asift_matches(this->num_of_tilts1, this->num_of_tilts1, rows1, cols1, rows2, cols2,
		0, this->keypoints1, this->keypoints2, matchings, siftparameters);

	for (int i = 0; i < matchings.size(); i++)
	{
		cv::DMatch dmatch;
		dmatch.distance = matchings[i].distance;
		dmatch.imgIdx = 0;
		dmatch.queryIdx = matchings[i].queryIdx;
		dmatch.trainIdx = matchings[i].trainIdx;
		matches.push_back(dmatch);
	}



	int feature_count1 = 0;	//特征图中特征的计数
	cv::Mat dst1(num_keys1, (int)VecLength, CV_32FC1, cv::Scalar::all(0));
	data1 = nullptr;
	for (int tt = 0; tt < (int)this->keypoints1.size(); tt++)
	{
		for (int rr = 0; rr < (int)this->keypoints1[tt].size(); rr++)
		{
			keypointslist::iterator ptr = this->keypoints1[tt][rr].begin();
			for (int i = 0; i < (int)this->keypoints1[tt][rr].size(); i++, ptr++)
			{
				cv::KeyPoint feature_point;
				feature_point.angle = ptr->angle;
				feature_point.pt.x = ptr->x;
				feature_point.pt.y = ptr->y;
				feature_point.size = ptr->scale;
				//cout << ptr->x << "  " << ptr->y << "  " << ptr->scale << "  " << ptr->angle;

				data1 = dst1.ptr<float>(feature_count1++);
				for (int ii = 0; ii < (int)VecLength; ii++)
				{
					*data1++ = ptr->vec[ii];
					//cout << "  " << ptr->vec[ii];
				}
				keypoint1.push_back(feature_point);
				//cout << std::endl;
			}
		}
	}

	descriptor1 = dst1;


	int feature_count2 = 0;	//特征图中特征的计数
	cv::Mat dst2(num_keys2, (int)VecLength, CV_32FC1, cv::Scalar::all(0));
	data2 = nullptr;
	for (int tt = 0; tt < (int)this->keypoints2.size(); tt++)
	{
		for (int rr = 0; rr < (int)this->keypoints2[tt].size(); rr++)
		{
			keypointslist::iterator ptr = this->keypoints2[tt][rr].begin();
			for (int i = 0; i < (int)this->keypoints2[tt][rr].size(); i++, ptr++)
			{
				cv::KeyPoint feature_point;
				feature_point.angle = ptr->angle;
				feature_point.pt.x = ptr->x;
				feature_point.pt.y = ptr->y;
				feature_point.size = ptr->scale;
				//cout << ptr->x << "  " << ptr->y << "  " << ptr->scale << "  " << ptr->angle;

				data2 = dst2.ptr<float>(feature_count2++);
				for (int ii = 0; ii < (int)VecLength; ii++)
				{
					*data2++ = ptr->vec[ii];
					//cout << "  " << ptr->vec[ii];
				}
				keypoint2.push_back(feature_point);
				//cout << std::endl;
			}
		}
	}

	descriptor2 = dst2;
}

