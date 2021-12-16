#include <opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <chrono>
#include <io.h>
#include <direct.h>
#include "sphere2cube.h"
#include "ASIFT_Detector.h"
#include "../../ransac.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <pcl/common/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace cv;
using namespace std;
using namespace xfeatures2d;

struct KeyPoint3D
{
    KeyPoint3D() : x(0.0), y(0.0), z(0.0) {}
    KeyPoint3D(Vec3d vec) : x(vec[0]), y(vec[1]), z(vec[2]) {}
    KeyPoint3D(double x, double y, double z) : x(x), y(y), z(z) {}
    double x, y, z;
};

struct Vec3dMatch
{
    Vec3dMatch() : key_points_match(), score(0.0) {}
    pair<KeyPoint3D, KeyPoint3D> key_points_match;
    double score;
};

Sphere2Cube s2c(1024);

void Pano2dToSphere3d(double theta, double phi, Vec3d &vec)
{
    vec[0] = 1.0 * sin(phi) * cos(theta);
    vec[1] = 1.0 * cos(phi);
    vec[2] = 1.0 * sin(phi) * sin(theta);
}

bool Cube2dToSphere3d(int tile_x, int tile_y, Vec3d &vec, int index, int tile_size)
{
    double theta = 0.0, phi = 0.0;
    switch (index)
    {
    case 0:
        tie(phi, theta) = s2c.func_up(tile_y, tile_x);
        Pano2dToSphere3d(theta, phi, vec);
        break;
    case 1:
        tie(phi, theta) = s2c.func_front(tile_y, tile_x);
        Pano2dToSphere3d(theta, phi, vec);
        break;
    case 2:
        tie(phi, theta) = s2c.func_right(tile_y, tile_x);
        Pano2dToSphere3d(theta, phi, vec);
        break;
    case 3:
        tie(phi, theta) = s2c.func_back(tile_y, tile_x);
        Pano2dToSphere3d(theta, phi, vec);
        break;
    case 4:
        tie(phi, theta) = s2c.func_left(tile_y, tile_x);
        Pano2dToSphere3d(theta, phi, vec);
        break;
    case 5:
        tie(phi, theta) = s2c.func_down(tile_y, tile_x);
        Pano2dToSphere3d(theta, phi, vec);
        break;
    default:
        return false;
        break;
    }
    return true;
}

std::vector<Mat> sphere2cube(Mat image, int tile_size = 1024)
{
    std::vector<Mat> result;
    Sphere2Cube s2c(tile_size);
    Faces cube;

    auto t1 = std::chrono::steady_clock::now();
    s2c.transform(image, cube);
    auto t2 = std::chrono::steady_clock::now();

    printf("Cost %f s.\n", std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count());

    for (int i = 0; i < 6; i++)
    {
        result.push_back(cube.faces[i]);
    }

    return result;
}

void r_asift_match(Mat src_img_1, Mat src_img_2, int tile_size = 1024)
{
    resize(src_img_1, src_img_1, Size(tile_size, tile_size));
    resize(src_img_2, src_img_2, Size(tile_size, tile_size));


    //imshow("src_img_1", src_img_1);
    //imshow("src_img_2", src_img_2);
    //waitKey(0);

    Mat gray_1, gray_2;
    cvtColor(src_img_1, gray_1, CV_BGR2GRAY);
    cvtColor(src_img_2, gray_2, CV_BGR2GRAY);


    gray_1.convertTo(gray_1, CV_32FC1);
    gray_2.convertTo(gray_2, CV_32FC1);

    //imshow("gray_1", gray_1);
    //imshow("gray_2", gray_2);
    //waitKey(0);

    CASIFT_Detector detector;
    vector<KeyPoint> keyPoint1, keyPoint2;
    Mat imageDesc1, imageDesc2;
    vector<DMatch> goodMatchesPoints;
    detector.DetectAndMatch(gray_1, gray_2, keyPoint1, keyPoint2, imageDesc1, imageDesc2, goodMatchesPoints);

    Mat output1, output2;
    drawKeypoints(src_img_1, keyPoint1, output1, cv::Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(src_img_2, keyPoint2, output2, cv::Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("Affine SIFT 1", output1);
    imshow("Affine SIFT 2", output2);
    waitKey(0);

    Mat good_match_img;
    drawMatches(src_img_1, keyPoint1, src_img_2, keyPoint2, goodMatchesPoints, good_match_img, cv::Scalar::all(-1), cv::Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("good_match_img ", good_match_img);
    waitKey(0);
}

void asift_match(Mat src_img_1, Mat src_img_2, int tile_size = 1024)
{
    resize(src_img_1, src_img_1, Size(tile_size, tile_size));
    resize(src_img_2, src_img_2, Size(tile_size, tile_size));


    //imshow("src_img_1", src_img_1);
    //imshow("src_img_2", src_img_2);
    //waitKey(0);

    Mat gray_1, gray_2;
    cvtColor(src_img_1, gray_1, CV_BGR2GRAY);
    cvtColor(src_img_2, gray_2, CV_BGR2GRAY);


    gray_1.convertTo(gray_1, CV_32FC1);
    gray_2.convertTo(gray_2, CV_32FC1);

    //imshow("gray_1", gray_1);
    //imshow("gray_2", gray_2);
    //waitKey(0);

    CASIFT_Detector detector;
    vector<KeyPoint> keyPoint1, keyPoint2;
    Mat imageDesc1, imageDesc2;
    detector.Detect(gray_1, keyPoint1, imageDesc1);
    detector.Detect(gray_2, keyPoint2, imageDesc2);

    Mat output1, output2;
    drawKeypoints(src_img_1, keyPoint1, output1, cv::Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(src_img_2, keyPoint2, output2, cv::Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("Affine SIFT 1", output1);
    imshow("Affine SIFT 2", output2);
    waitKey(0);

    FlannBasedMatcher matcher;
    vector<vector<DMatch>> matchesPoints;
    vector<DMatch> goodMatchesPoints;

    vector<Mat> train_desc(1, imageDesc1);
    matcher.add(train_desc);
    matcher.train();

    matcher.knnMatch(imageDesc2, matchesPoints, 2);
    cout << "total match points: " << matchesPoints.size() << endl;

    for (int i = 0; i < matchesPoints.size(); i++)
    {
        if (matchesPoints[i][0].distance < 0.5 * matchesPoints[i][1].distance)
        {
            goodMatchesPoints.push_back(matchesPoints[i][0]);
        }
    }

    Mat good_match_img;
    drawMatches(src_img_2, keyPoint2, src_img_1, keyPoint1, goodMatchesPoints, good_match_img);
    imshow("good_match_img ", good_match_img);
    waitKey(0);
}

vector<Vec3dMatch> sift_match(Mat src_img_1, Mat src_img_2, int index_1, int index_2, int tile_size=1024)
{
#define SHOW_IMG 0
    resize(src_img_1, src_img_1, Size(tile_size, tile_size));
    resize(src_img_2, src_img_2, Size(tile_size, tile_size));


    //imshow("src_img_1", src_img_1);
    //imshow("src_img_2", src_img_2);
    //waitKey(0);

#if 0
    Mat gray_1, gray_2;
    cvtColor(src_img_1, gray_1, CV_BGR2GRAY);
    cvtColor(src_img_2, gray_2, CV_BGR2GRAY);
#else
    Mat gray_1, gray_2;
    gray_1 = src_img_1;
    gray_2 = src_img_2;
#endif


    //gray_1.convertTo(gray_1, CV_32FC1);
    //gray_2.convertTo(gray_2, CV_32FC1);

    //imshow("gray_1", gray_1);
    //imshow("gray_2", gray_2);
    //waitKey(0);

    Ptr<SIFT> sift = xfeatures2d::SIFT::create();
    vector<KeyPoint> keyPoint1, keyPoint2;
    sift->detect(gray_1, keyPoint1);
    sift->detect(gray_2, keyPoint2);

    //特征点描述，为下边的特征点匹配做准备
    Mat imageDesc1, imageDesc2;
    sift->compute(gray_1, keyPoint1, imageDesc1);
    sift->compute(gray_2, keyPoint2, imageDesc2);

    //Mat output1, output2;
    //drawKeypoints(src_img_1, keyPoint1, output1, cv::Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    //drawKeypoints(src_img_2, keyPoint2, output2, cv::Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    //imshow("Affine SIFT 1", output1);
    //imshow("Affine SIFT 2", output2);
    //waitKey(0);

    FlannBasedMatcher matcher;
    vector<vector<DMatch>> matchesPoints;
    vector<DMatch> goodMatchesPoints;

    vector<Mat> train_desc(1, imageDesc1);
    matcher.add(train_desc);
    matcher.train();

    matcher.knnMatch(imageDesc2, matchesPoints, 2);
    cout << "total match points: " << matchesPoints.size() << endl;

    for (int i = 0; i < matchesPoints.size(); i++)
    {
        if (matchesPoints[i][0].distance < 0.6 * matchesPoints[i][1].distance)
        {
            goodMatchesPoints.push_back(matchesPoints[i][0]);
        }
    }

    Mat good_match_img;
    drawMatches(src_img_2, keyPoint2, src_img_1, keyPoint1, goodMatchesPoints, good_match_img, cv::Scalar::all(-1), cv::Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

#if SHOW_IMG == 1
    imshow("good_match_img ", good_match_img);
    waitKey(0);
#endif

    vector<Point2f> imagePoints1, imagePoints2;

    for (int i = 0; i < goodMatchesPoints.size(); i++)
    {
        imagePoints2.push_back(keyPoint2[goodMatchesPoints[i].queryIdx].pt);
        imagePoints1.push_back(keyPoint1[goodMatchesPoints[i].trainIdx].pt);
    }

    vector<Vec3dMatch> matching_list_3d;
    matching_list_3d.reserve(goodMatchesPoints.size());
    for (int k = 0; k < goodMatchesPoints.size(); k++)
    {
        Vec3d vec1, vec2;
        Cube2dToSphere3d(imagePoints1[k].x, imagePoints1[k].y, vec1, index_1, tile_size);
        Cube2dToSphere3d(imagePoints2[k].x, imagePoints2[k].y, vec2, index_2, tile_size);
        Vec3dMatch m;
        m.key_points_match = pair<KeyPoint3D, KeyPoint3D>(KeyPoint3D(vec1), KeyPoint3D(vec2));
        m.score = 1.0;
        matching_list_3d.push_back(std::move(m));
    }
#if SHOW_IMG == 1
    waitKey(0);
#endif
    return matching_list_3d;
}



vector<Vec3dMatch> surf_match(Mat src_img_1, Mat src_img_2, int index_1, int index_2, int tile_size = 1024)
{

    resize(src_img_1, src_img_1, Size(tile_size, tile_size));
    resize(src_img_2, src_img_2, Size(tile_size, tile_size));


    //imshow("src_img_1", src_img_1);
    //imshow("src_img_2", src_img_2);
    //waitKey(0);

#if 0
    Mat gray_1, gray_2;
    cvtColor(src_img_1, gray_1, CV_BGR2GRAY);
    cvtColor(src_img_2, gray_2, CV_BGR2GRAY);
#else
    Mat gray_1, gray_2;
    gray_1 = src_img_1;
    gray_2 = src_img_2;
#endif


    //gray_1.convertTo(gray_1, CV_32FC1);
    //gray_2.convertTo(gray_2, CV_32FC1);

    //imshow("gray_1", gray_1);
    //imshow("gray_2", gray_2);
    //waitKey(0);

    Ptr<SURF> surf = xfeatures2d::SURF::create(400);
    vector<KeyPoint> keyPoint1, keyPoint2;
    surf->detect(gray_1, keyPoint1);
    surf->detect(gray_2, keyPoint2);

    //特征点描述，为下边的特征点匹配做准备
    Mat imageDesc1, imageDesc2;
    surf->compute(gray_1, keyPoint1, imageDesc1);
    surf->compute(gray_2, keyPoint2, imageDesc2);

    Mat output1, output2;
    drawKeypoints(src_img_1, keyPoint1, output1, cv::Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(src_img_2, keyPoint2, output2, cv::Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("Affine SIFT 1", output1);
    imshow("Affine SIFT 2", output2);
    waitKey(0);

    FlannBasedMatcher matcher;
    vector<vector<DMatch>> matchesPoints;
    vector<DMatch> goodMatchesPoints;

    vector<Mat> train_desc(1, imageDesc1);
    matcher.add(train_desc);
    matcher.train();

    matcher.knnMatch(imageDesc2, matchesPoints, 2);
    cout << "total match points: " << matchesPoints.size() << endl;

    for (int i = 0; i < matchesPoints.size(); i++)
    {
        if (matchesPoints[i][0].distance < 0.5 * matchesPoints[i][1].distance)
        {
            goodMatchesPoints.push_back(matchesPoints[i][0]);
        }
    }

    Mat good_match_img;
    drawMatches(src_img_2, keyPoint2, src_img_1, keyPoint1, goodMatchesPoints, good_match_img, cv::Scalar::all(-1), cv::Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    imshow("good_match_img ", good_match_img);
    waitKey(0);

    vector<Point2f> imagePoints1, imagePoints2;

    for (int i = 0; i < goodMatchesPoints.size(); i++)
    {
        imagePoints2.push_back(keyPoint2[goodMatchesPoints[i].queryIdx].pt);
        imagePoints1.push_back(keyPoint1[goodMatchesPoints[i].trainIdx].pt);
    }

    vector<Vec3dMatch> matching_list_3d;
    matching_list_3d.reserve(goodMatchesPoints.size());
    for (int k = 0; k < goodMatchesPoints.size(); k++)
    {
        Vec3d vec1, vec2;
        Cube2dToSphere3d(imagePoints1[k].x, imagePoints1[k].y, vec1, index_1, tile_size);
        Cube2dToSphere3d(imagePoints2[k].x, imagePoints2[k].y, vec2, index_2, tile_size);
        Vec3dMatch m;
        m.key_points_match = pair<KeyPoint3D, KeyPoint3D>(KeyPoint3D(vec1), KeyPoint3D(vec2));
        m.score = 1.0;
        matching_list_3d.push_back(std::move(m));
    }
    waitKey(0);
    return matching_list_3d;
}

void orb_match(Mat src_img_1, Mat src_img_2, int tile_size = 1024)
{

    resize(src_img_1, src_img_1, Size(tile_size, tile_size));
    resize(src_img_2, src_img_2, Size(tile_size, tile_size));


    //imshow("src_img_1", src_img_1);
    //imshow("src_img_2", src_img_2);
    //waitKey(0);

    Mat gray_1, gray_2;
    cvtColor(src_img_1, gray_1, CV_BGR2GRAY);
    cvtColor(src_img_2, gray_2, CV_BGR2GRAY);


    //gray_1.convertTo(gray_1, CV_32FC1);
    //gray_2.convertTo(gray_2, CV_32FC1);

    //imshow("gray_1", gray_1);
    //imshow("gray_2", gray_2);
    //waitKey(0);

    Ptr<ORB> orb = ORB::create();
    vector<KeyPoint> keyPoint1, keyPoint2;
    orb->detect(gray_1, keyPoint1);
    orb->detect(gray_2, keyPoint2);

    //特征点描述，为下边的特征点匹配做准备
    Mat imageDesc1, imageDesc2;
    orb->compute(gray_1, keyPoint1, imageDesc1);
    orb->compute(gray_2, keyPoint2, imageDesc2);

    Mat output1, output2;
    drawKeypoints(src_img_1, keyPoint1, output1, cv::Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(src_img_2, keyPoint2, output2, cv::Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("Affine SIFT 1", output1);
    imshow("Affine SIFT 2", output2);
    waitKey(0);

    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matchesPoints;
    vector<DMatch> goodMatchesPoints;

    matcher.match(imageDesc1, imageDesc2, matchesPoints);
    cout << "total match points: " << matchesPoints.size() << endl;

    double min_dist = 9999999;
    for (int i = 0; i < matchesPoints.size(); i++)
    {
        if (matchesPoints[i].distance < min_dist)
        {
            min_dist = matchesPoints[i].distance;
        }
    }

    for (int i = 0; i < matchesPoints.size(); i++)
    {
        if (matchesPoints[i].distance <= max(2 * min_dist, 30.0))
        {
            goodMatchesPoints.push_back(matchesPoints[i]);
        }
    }

    Mat good_match_img;
    drawMatches(src_img_1, keyPoint1, src_img_2, keyPoint2, goodMatchesPoints, good_match_img);
    imshow("good_match_img", good_match_img);

    vector<Point2f> imagePoints1, imagePoints2;

    for (int i = 0; i < goodMatchesPoints.size(); i++)
    {
        imagePoints2.push_back(keyPoint2[goodMatchesPoints[i].queryIdx].pt);
        imagePoints1.push_back(keyPoint1[goodMatchesPoints[i].trainIdx].pt);
    }

    //GaussianBlur(stitchedImage, stitchedImage, Size(9, 9), 0.0);
    //medianBlur(stitchedImage, stitchedImage, 5);
    waitKey(0);
}


void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void* viewer_void)
{
    static int count = 0;
    pcl::visualization::PCLVisualizer* viewer = static_cast<pcl::visualization::PCLVisualizer*> (viewer_void);
    if (event.getKeySym() == "w" && event.keyDown())
    {
        count++;
        viewer->setCameraPosition(0, 0, count * 0.1, 0, 0, 0, 0, 0, 1);
    }
    if (event.getKeySym() == "s" && event.keyDown())
    {
        count--;
        viewer->setCameraPosition(0, 0, count * 0.1, 0, 0, 0, 0, 0, 1);
    }
}



int main(int argc, char** argv) {
    String base_url = "C:/Users/Albert/Desktop/114/";
    String url1 = "1";
    String url2 = "2";
    Mat image1 = cv::imread(base_url + url1 + ".jpg", CV_LOAD_IMAGE_COLOR);
    Mat image2 = cv::imread(base_url + url2 + ".jpg", CV_LOAD_IMAGE_COLOR);

    String url3 = "Depth1";
    String url4 = "Depth2";
    Mat image3 = cv::imread(base_url + url3 + ".jpg", -1);
    Mat image4 = cv::imread(base_url + url4 + ".jpg", -1);
    image3.rowRange(0, 100) = 0;
    image3.rowRange(444, 512) = 0;
    image4.rowRange(0, 100) = 0;
    image4.rowRange(444, 512) = 0;

    GaussianBlur(image1, image1, Size(3, 3), 0.0);
    GaussianBlur(image2, image2, Size(3, 3), 0.0);
    vector<Mat> result1 = sphere2cube(image1, 1024);
    vector<Mat> result2 = sphere2cube(image2, 1024);

    //for (int i = 0; i < min(result1.size(), result2.size()); i++)
    //{
    //    imshow("1_" + to_string(i), result1[i]);
    //    imshow("2_" + to_string(i), result2[i]);
    //}
    
    vector<Vec3dMatch> all_matches;
    for (int i = 0; i < result1.size(); i++)
    {
        for (int j = i; j < result2.size(); j++)
        {
            vector<Vec3dMatch> dm = sift_match(result1[i], result2[j], i, j, 1024);
            all_matches.insert(all_matches.end(), dm.begin(), dm.end());

        }
    }
    /*
    Mat points1 = (Mat_<double>(10, 3) << sqrt(2) / 2, 0, sqrt(2) / 2, 0.5, 0, sqrt(3) / 2,
        -sqrt(2) / 2, 0, sqrt(2) / 2, 0, 0, -1, -sqrt(2) / 2, sqrt(2) / 2, 0,
        -sqrt(3) / 3,  -sqrt(3) / 3, -sqrt(3) / 3, -sqrt(2) / 2, sqrt(2) / 2, 0, 0, 0, 1, 0, sqrt(2) / 2, sqrt(2) / 2, 0, sqrt(2), sqrt(2));
    Mat points2 = (Mat_<double>(10, 3) << 0.5, 0, sqrt(3) / 2, -0.5, 0, sqrt(3) / 2,
        sqrt(2) / 2, 0, sqrt(2) / 2, sqrt(3) / 2, 0, 0.5, sqrt(6) / 3, sqrt(6) / 6, sqrt(6) / 6,
        3.0 * sqrt(11) / 11, -sqrt(11) / 11, sqrt(11) / 11, sqrt(2) / 2, sqrt(2) / 2, 0, 0, 0, 1, 0, sqrt(2) / 2, sqrt(2) / 2, 0, sqrt(2), sqrt(2));

    //Mat points1 = (Mat_<double>(5, 3) << 0, 0, 1, sqrt(2) / 2, 0, sqrt(2) / 2, 0.5, 0, sqrt(3) / 2, -sqrt(2) / 2, 0, sqrt(2) / 2, 0, 0, -1);
    //Mat points2 = (Mat_<double>(5, 3) << 0, 0, 1, 0.5, 0, sqrt(3) / 2, -0.5, 0, sqrt(3) / 2, sqrt(2) / 2, 0, sqrt(2) / 2, 0, sqrt(3) / 2, 0.5);
 
    cout << points1 << endl;
    cout << points2 << endl;
    */
    /*
    Mat E, R, t;
    vector<Mat> Inliers;
    tie(E, Inliers) = cvv::findEssentialMat(points1, points2);
    cout << "Inliers" << Inliers[0] << endl << endl << Inliers[1] << endl;
    cvv::recoverPose(E, Inliers[0], Inliers[1], R, t);
    cout << R << endl;
    cout << t << endl;
    */
    Mat points1, points2;
    double x, y, z;
    points1.create(all_matches.size(), 3, CV_64FC1);
    points2.create(all_matches.size(), 3, CV_64FC1);
    for (int i = 0; i < all_matches.size(); i++)
    {
        auto ptr1 = points1.ptr<double>(i);
        auto ptr2 = points2.ptr<double>(i);
        auto m = all_matches[i];
        x = m.key_points_match.first.x;
        y = m.key_points_match.first.y;
        z = m.key_points_match.first.z;
        ptr1[0] = x; ptr1[1] = y; ptr1[2] = z; 
        x = m.key_points_match.second.x;
        y = m.key_points_match.second.y;
        z = m.key_points_match.second.z;
        ptr2[0] = x; ptr2[1] = y; ptr2[2] = z;
    }
    
    Mat E, R, t;
    vector<Mat> Inliers;

    tie(E, Inliers) = cvv::findEssentialMat(points1, points2);
    cout << "Inliers" << Inliers[0] << endl << endl << Inliers[1] << endl;
    cvv::recoverPose(E, Inliers[0], Inliers[1], R, t);
    cout << R << endl;
    cout << t << endl;

    Eigen::Isometry3d T;
    double ratio = -1.0;
    T(0, 3) = t.at<double>(0, 0) * ratio;
    T(1, 3) = t.at<double>(1, 0) * ratio;
    T(2, 3) = t.at<double>(2, 0) * ratio;
    T(3, 3) = 1.0;

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            T(i, j) = R.at<double>(i, j);
        }
    }

    int rows = image3.rows;
    int cols = image3.cols;
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
    PointCloud::Ptr pc(new PointCloud);
    for (int v = 0; v < rows; v++)
    {
        for (int u = 0; u < cols; u++)
        {
            int vv = int((double)v / rows * image1.rows);
            int uu = int((double)u / cols * image1.cols);

            unsigned char d = image3.ptr<uchar>(v, u)[0];
            if (d == 0) continue;
            double depth = d / 255.0 * 10.0;

            Eigen::Vector3d point;
            double phi = (double)v / rows * PI;
            double theta = (double)u / cols * 2 * PI;
            point[0] = sin(phi) * cos(theta) * depth;
            point[1] = cos(phi) * depth;
            point[2] = sin(phi) * sin(theta) * depth;

            Eigen::Vector3d point_world = T * point;
            PointT p;
            p.x = point_world[0];
            p.y = point_world[1];
            p.z = point_world[2];
            p.b = image1.ptr<uchar>(vv, uu)[0];
            p.g = image1.ptr<uchar>(vv, uu)[1];
            p.r = image1.ptr<uchar>(vv, uu)[2];
            pc->points.push_back(p);
        }
    }
    for (int v = 0; v < rows; v++)
    {
        for (int u = 0; u < cols; u++)
        {
            int vv = int((double)v / rows * image2.rows);
            int uu = int((double)u / cols * image2.cols);

            unsigned char d = image4.ptr<uchar>(v, u)[0];
            if (d == 0) continue;
            double depth = d / 255.0 * 10.0;

            Eigen::Vector3d point;
            double phi = (double)v / rows * PI;
            double theta = (double)u / cols * 2 * PI;

            point[0] = sin(phi) * cos(theta) * depth;
            point[1] = cos(phi) * depth;
            point[2] = sin(phi) * sin(theta) * depth;

            PointT p;
            p.x = point[0];
            p.y = point[1];
            p.z = point[2];
            p.b = image2.ptr<uchar>(vv, uu)[0];
            p.g = image2.ptr<uchar>(vv, uu)[1];
            p.r = image2.ptr<uchar>(vv, uu)[2];
            pc->points.push_back(p);
        }
    }


    pc->is_dense = false;
    cout << "点云个数" << pc->size() << endl;
    pcl::io::savePCDFileBinary(base_url + "map.pcd", *pc);



    //pcl::visualization::PCLVisualizer viewer("viewer");
    //viewer.addCoordinateSystem(1.0, "reference");
    //viewer.setCameraPosition(0, 0, 0, 0, 0, 0, 0, 0, 1);
    //viewer.addPointCloud<PointT>(pc);


    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud<PointT>(pc, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer->addCoordinateSystem(1.0, "reference");
    viewer->setCameraPosition(0, 0, 0, 0, 0, 0, 0, 0, 1);
    viewer->initCameraParameters();
    viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)viewer.get());


    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        //boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }

    


    /*
    Mat rot_v;
    Mat rot_m = (cv::Mat_<double>(3, 3) << 0.70710678, 0, 0.70710678, 0.270598, -0.9238795, -0.270598, 0.65328148, 0.382683, -0.65328148);
    cv::Rodrigues(rot_m, rot_v);
    double angle = norm(rot_v);
    rot_v.col(0) = rot_v.col(0) / angle;
    cout << rot_v << "-" << angle * 180.0 / 3.1415926 << endl;
    */

    /*
    Mat P0 = Mat::eye(3, 4, CV_64F);
    Mat P1 = (Mat_<double>(3, 4) << 0, 0, 1, -sqrt(2), 0, 1, 0, 0, -1, 0, 0, sqrt(2));
    Mat point1 = (Mat_<double>(2, 1) << 0, 0);
    Mat point2 = (Mat_<double>(2, 1) << 0, 0);
    Mat point3 = (Mat_<double>(3, 1) << sqrt(2), 0, sqrt(2));
    Mat result;
    triangulatePoints(P0, P1, point1, point2, result);
    cout << result.at<double>(0, 0) / result.at<double>(3, 0) << "-" << result.at<double>(1, 0) / result.at<double>(3, 0) << "-" << result.at<double>(2, 0) / result.at<double>(3, 0) << "-" << result.at<double>(3, 0) / result.at<double>(3, 0) << "-" << endl;
    Mat R = P1.colRange(0, 3);
    Mat t = P1.col(3);
    cout << R * point3 + t << endl;
    */

    waitKey(0);
    return 0;
}
