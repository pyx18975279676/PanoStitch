//#include "ASIFT_Detector.h"
//#include <opencv.hpp>
//#include <opencv2/xfeatures2d.hpp>
//
//using namespace cv;
//using namespace std;
//using namespace xfeatures2d;
//
//typedef struct
//{
//    Point2f left_top;
//    Point2f left_bottom;
//    Point2f right_top;
//    Point2f right_bottom;
//}four_corners_t;
//
//four_corners_t corners;
//
//void CalcCorners(const Mat& H, const Mat& src)
//{
//    double v2[] = { 0, 0, 1 };//左上角
//    double v1[3];//变换后的坐标值
//    Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
//    Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
//
//    V1 = H * V2;
//    //左上角(0,0,1)
//    cout << "V2: " << V2 << endl;
//    cout << "V1: " << V1 << endl;
//    corners.left_top.x = v1[0] / v1[2];
//    corners.left_top.y = v1[1] / v1[2];
//
//    //左下角(0,src.rows,1)
//    v2[0] = 0;
//    v2[1] = src.rows;
//    v2[2] = 1;
//    V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
//    V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
//    V1 = H * V2;
//    corners.left_bottom.x = v1[0] / v1[2];
//    corners.left_bottom.y = v1[1] / v1[2];
//
//    //右上角(src.cols,0,1)
//    v2[0] = src.cols;
//    v2[1] = 0;
//    v2[2] = 1;
//    V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
//    V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
//    V1 = H * V2;
//    corners.right_top.x = v1[0] / v1[2];
//    corners.right_top.y = v1[1] / v1[2];
//
//    //右下角(src.cols,src.rows,1)
//    v2[0] = src.cols;
//    v2[1] = src.rows;
//    v2[2] = 1;
//    V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
//    V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
//    V1 = H * V2;
//    corners.right_bottom.x = v1[0] / v1[2];
//    corners.right_bottom.y = v1[1] / v1[2];
//
//}
//
////优化两图的连接处，使得拼接自然
//void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst)
//{
//    int start = MIN(corners.left_top.x, corners.left_bottom.x);//开始位置，即重叠区域的左边界  
//
//    double processWidth = img1.cols - start;//重叠区域的宽度  
//    int rows = dst.rows;
//    int cols = img1.cols; //注意，是列数*通道数
//    double alpha = 1;//img1中像素的权重  
//    for (int i = 0; i < rows; i++)
//    {
//        uchar* p = img1.ptr<uchar>(i);  //获取第i行的首地址
//        uchar* t = trans.ptr<uchar>(i);
//        uchar* d = dst.ptr<uchar>(i);
//        for (int j = start; j < cols; j++)
//        {
//            //如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
//            if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
//            {
//                alpha = 1;
//            }
//            else
//            {
//                //img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好  
//                alpha = (processWidth - (j - start)) / processWidth;
//            }
//
//            d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
//            d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
//            d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);
//
//        }
//    }
//
//}
//
//int main(int argc, char** argv) {
//    Mat src_img_1, src_img_2;
//    src_img_1 = imread("C:/Users/Albert/Desktop/113/4.jpg", CV_LOAD_IMAGE_COLOR);
//    src_img_2 = imread("C:/Users/Albert/Desktop/113/3.jpg", CV_LOAD_IMAGE_COLOR);
//
//    resize(src_img_1, src_img_1, Size(src_img_1.cols / 2, src_img_1.rows / 2));
//    resize(src_img_2, src_img_2, Size(src_img_2.cols / 2, src_img_2.rows / 2));
//
//
//    imshow("src_img_1", src_img_1);
//    imshow("src_img_2", src_img_2);
//    waitKey(0);
//
//    Mat gray_1, gray_2;
//    cvtColor(src_img_1, gray_1, CV_BGR2GRAY);
//    cvtColor(src_img_2, gray_2, CV_BGR2GRAY);
//
//
//    gray_1.convertTo(gray_1, CV_32FC1);
//    gray_2.convertTo(gray_2, CV_32FC1);
//
//    imshow("gray_1", gray_1);
//    imshow("gray_2", gray_2);
//    waitKey(0);
//
//    //Ptr<SIFT> sift = xfeatures2d::SIFT::create();
//    //vector<KeyPoint> keyPoint1, keyPoint2;
//    //sift->detect(gray_1, keyPoint1);
//    //sift->detect(gray_2, keyPoint2);
//
//    //特征点描述，为下边的特征点匹配做准备
//    //Mat imageDesc1, imageDesc2;
//    //sift->compute(gray_1, keyPoint1, imageDesc1);
//    //sift->compute(gray_2, keyPoint2, imageDesc2);
//
//    CASIFT_Detector detector;
//    vector<KeyPoint> keyPoint1, keyPoint2;
//    Mat imageDesc1, imageDesc2;
//    detector.Detect(gray_1, keyPoint1, imageDesc1);
//    detector.Detect(gray_2, keyPoint2, imageDesc2);
//
//    Mat output1, output2;
//    drawKeypoints(src_img_1, keyPoint1, output1, cv::Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//    drawKeypoints(src_img_2, keyPoint2, output2, cv::Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//    imshow("Affine SIFT 1", output1);
//    imshow("Affine SIFT 2", output2);
//    waitKey(0);
//
//    FlannBasedMatcher matcher;
//    vector<vector<DMatch>> matchesPoints;
//    vector<DMatch> goodMatchesPoints;
//
//    vector<Mat> train_desc(1, imageDesc1);
//    matcher.add(train_desc);
//    matcher.train();
//
//    matcher.knnMatch(imageDesc2, matchesPoints, 2);
//    cout << "total match points: " << matchesPoints.size() << endl;
//
//    for (int i = 0; i < matchesPoints.size(); i++)
//    {
//        if (matchesPoints[i][0].distance < 0.5 * matchesPoints[i][1].distance)
//        {
//            goodMatchesPoints.push_back(matchesPoints[i][0]);
//        }
//    }
//
//    Mat good_match;
//    drawMatches(src_img_2, keyPoint2, src_img_1, keyPoint1, goodMatchesPoints, good_match);
//    imshow("good_match ", good_match);
//
//    vector<Point2f> imagePoints1, imagePoints2;
//
//    for (int i = 0; i < goodMatchesPoints.size(); i++)
//    {
//        imagePoints2.push_back(keyPoint2[goodMatchesPoints[i].queryIdx].pt);
//        imagePoints1.push_back(keyPoint1[goodMatchesPoints[i].trainIdx].pt);
//    }
//
//    Mat H = findHomography(imagePoints1, imagePoints2, CV_RANSAC);
//
//    CalcCorners(H, src_img_1);
//
//    Mat imageTransform1, imageTransform2;
//    warpPerspective(src_img_1, imageTransform1, H, Size(max(corners.right_top.x, corners.right_bottom.x), src_img_2.rows));
//    //warpPerspective(image01, imageTransform2, adjustMat*homo, Size(image02.cols*1.3, image02.rows*1.8));
//    imshow("123", imageTransform1);
//    waitKey(0);
//
//
//    int dst_width = imageTransform1.cols;  //取最右点的长度为拼接图的长度
//    int dst_height = src_img_2.rows;
//
//    Mat stitchedImage(dst_height, dst_width, CV_8UC3);
//    stitchedImage.setTo(0);
//
//    imageTransform1.copyTo(stitchedImage(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
//    src_img_2.copyTo(stitchedImage(Rect(0, 0, src_img_2.cols, src_img_2.rows)));
//
//    OptimizeSeam(src_img_2, imageTransform1, stitchedImage);
//
//    //GaussianBlur(stitchedImage, stitchedImage, Size(9, 9), 0.0);
//    //medianBlur(stitchedImage, stitchedImage, 5);
//
//    imshow("stitchedImage", stitchedImage);
//    waitKey(0);
//
//    waitKey(0);
//    return 0;
//}
