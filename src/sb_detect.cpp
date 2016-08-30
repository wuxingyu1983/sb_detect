/*
 * sb_detect.cpp
 *
 *  Created on: 2016年8月30日
 *      Author: wuxingyu
 */


#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace cv::xfeatures2d;

void readme();
void edge_detect(OutputArray dst);

int main(int argc, char** argv )
{
	if ( argc != 3 && argc != 4)
	{
		readme();
		return -1;
	}

	int debug = 0;
	if (argc == 4) {
		debug = 1;
	}

	Mat img_scene, img_object;
	img_scene = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
	// 目标图片
	img_object = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );

	if ( !img_scene.data )
	{
		std::cout << "No Scene Image \n" << std::endl;
		return -1;
	}

	if ( !img_object.data )
	{
		std::cout << "No Object Image \n" << std::endl;
		return -1;
	}

	edge_detect(img_scene);
	edge_detect(img_object);

	std::vector<KeyPoint> keypoints_object, keypoints_scene;

	Ptr<SIFT> scene_detector = SIFT::create( 10000 );
	scene_detector->detect( img_scene, keypoints_scene );

	Ptr<SIFT> object_detector = SIFT::create( 400 );
	object_detector->detect( img_object, keypoints_object );

	if (debug) {
		/*
		   Mat img_keypoints_scene, img_keypoints_object;
		   drawKeypoints( img_object, keypoints_object, img_keypoints_object, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
		   drawKeypoints( img_scene, keypoints_scene, img_keypoints_scene, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

		   namedWindow("Display Image", WINDOW_NORMAL );
		   imshow("Display Image", img_keypoints_scene);

		   namedWindow("Target Image", WINDOW_NORMAL );
		   imshow("Target Image", img_keypoints_object);
		 */
	}

	//-- Step 2: Calculate descriptors (feature vectors)
	Ptr<SiftDescriptorExtractor> extractor = SiftDescriptorExtractor::create(10000);

	Mat descriptors_object, descriptors_scene;

	extractor->compute( img_object, keypoints_object, descriptors_object );
	extractor->compute( img_scene, keypoints_scene, descriptors_scene );

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	BFMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match( descriptors_object, descriptors_scene, matches );

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for( int i = 0; i < descriptors_object.rows; i++ )
	{ double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}

	if (debug) {
		printf("-- Max dist : %f \n", max_dist );
		printf("-- Min dist : %f \n", min_dist );
	}

	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector< DMatch > good_matches;

	for( int i = 0; i < descriptors_object.rows; i++ )
	{
		if( matches[i].distance < 3*min_dist )
		{ good_matches.push_back( matches[i]); }
	}

	Mat img_matches;
	drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
			good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
			std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	if (debug) {
		printf("the size of good_matches is %d\n", good_matches.size());
	}

	for( int i = 0; i < good_matches.size(); i++ )
	{
		//-- Get the keypoints from the good matches
		obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
		scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
	}

	Mat H = findHomography( obj, scene, CV_RANSAC );

	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
	obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
	std::vector<Point2f> scene_corners(4);

	perspectiveTransform( obj_corners, scene_corners, H);

	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
	line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
	line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
	line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );

	Point2f center_point = (scene_corners[0] + scene_corners[1] + scene_corners[2] + scene_corners[3]) / 4 + Point2f( img_object.cols, 0);
	circle( img_matches, center_point, 10, Scalar(0, 255, 0), 5);
	printf("%f %f\n", center_point.x, center_point.y);

	if (debug) {
		//-- Show detected matches
		namedWindow("Good Matches & Object detection", WINDOW_NORMAL );
		imshow( "Good Matches & Object detection", img_matches );
		waitKey(0);
	}

	return 0;
}

/** @function readme */
void readme()
{
	std::cout << "usage: sb_detect <Scene_Path> <Object_Path> [debug]\n" << std::endl;
}

void edge_detect(OutputArray dst) {
	// laplace edge detect
	int kernel_size = 3;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	Mat abs_dst;

	/// Remove noise by blurring with a Gaussian filter
	GaussianBlur( dst, dst, Size(3,3), 0, 0, BORDER_DEFAULT );

	/// Apply Laplace function
	Laplacian( dst, abs_dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( abs_dst, dst );
}

