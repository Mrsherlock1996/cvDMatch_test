#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
	Mat obj = imread("1.jpg");   //载入目标图像
	Mat scene = imread("2.jpg"); //载入场景图像
	if (obj.empty() || scene.empty())
	{
		cout << "Can't open the picture!\n";
		return 0;
	}
	vector<KeyPoint> obj_keypoints, scene_keypoints;
	Mat obj_descriptors, scene_descriptors;
	//ORB detector;     //采用ORB算法提取特征点
	//ORB orb;
	Ptr<ORB> detector = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
	detector->detect(obj, obj_keypoints);
	detector->detect(scene, scene_keypoints);
	detector->compute(obj, obj_keypoints, obj_descriptors);
	detector->compute(scene, scene_keypoints, scene_descriptors);
	BFMatcher matcher(NORM_HAMMING, true); //汉明距离做为相似度度量

	//匹配结果放在matches里面
	vector<DMatch> matches;
	matcher.match(obj_descriptors, scene_descriptors, matches);//匹配成功的点对数为257,

	//按照匹配关系将图画出来，背景图为match_img
	Mat match_img;
	drawMatches(obj, obj_keypoints, scene, scene_keypoints, matches, match_img);
	imshow("滤除误匹配前", match_img);

	//保存匹配对序号
	vector<int> queryIdxs(matches.size()), trainIdxs(matches.size());
	for (size_t i = 0; i < matches.size(); i++)
	{
		queryIdxs[i] = matches[i].queryIdx;//取出查询图片中匹配的点对的索引即id号；那么queryIdxs、trainIdxs都为257
		trainIdxs[i] = matches[i].trainIdx;//取出训练图片中匹配的点对的索引即id号；
	}

	Mat H12;   //变换矩阵

	vector<Point2f> points1;
	KeyPoint::convert(obj_keypoints, points1, queryIdxs);//KeyPoint根据索引转point2f坐标
	vector<Point2f> points2;
	KeyPoint::convert(scene_keypoints, points2, trainIdxs);//同上
	int ransacReprojThreshold = 3;  //拒绝阈值

	cout << "matches size = " << matches.size() << endl;
	H12 = findHomography(Mat(points1), Mat(points2), CV_RANSAC, ransacReprojThreshold);//计算单应矩阵
	vector<char> matchesMask(matches.size(), 0);//用于内点标记使用
	Mat points1t;
	perspectiveTransform(Mat(points1), points1t, H12);//对输入二维点做透视变换，变换后的结果保存在points1t中
	for (size_t i1 = 0; i1 < points1.size(); i1++)  //保存‘内点’
	{
		if (norm(points2[i1] - points1t.at<Point2f>((int)i1, 0)) <= ransacReprojThreshold) //给内点做标记
		{
			matchesMask[i1] = 1;
		}
	}
	Mat match_img2;   //滤除‘外点’后
	drawMatches(obj, obj_keypoints, scene, scene_keypoints, matches, match_img2, Scalar(0, 0, 255), Scalar::all(-1), matchesMask);

	//画出目标位置
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(obj.cols, 0);
	obj_corners[2] = cvPoint(obj.cols, obj.rows); obj_corners[3] = cvPoint(0, obj.rows);
	std::vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, H12);
	line(match_img2, scene_corners[0] + Point2f(static_cast<float>(obj.cols), 0),
		scene_corners[1] + Point2f(static_cast<float>(obj.cols), 0), Scalar(0, 0, 255), 2);
	line(match_img2, scene_corners[1] + Point2f(static_cast<float>(obj.cols), 0),
		scene_corners[2] + Point2f(static_cast<float>(obj.cols), 0), Scalar(0, 0, 255), 2);
	line(match_img2, scene_corners[2] + Point2f(static_cast<float>(obj.cols), 0),
		scene_corners[3] + Point2f(static_cast<float>(obj.cols), 0), Scalar(0, 0, 255), 2);
	line(match_img2, scene_corners[3] + Point2f(static_cast<float>(obj.cols), 0),
		scene_corners[0] + Point2f(static_cast<float>(obj.cols), 0), Scalar(0, 0, 255), 2);

	imshow("滤除误匹配后", match_img2);
	waitKey(0);

	return 0;
}

