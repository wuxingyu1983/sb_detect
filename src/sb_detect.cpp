/*
 * sb_detect.cpp
 *
 *  Created on: 2016年8月30日
 *      Author: wuxingyu
 */


#include <stdio.h>
#include <string.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace cv::xfeatures2d;

void readme();
void edge_detect(OutputArray dst);
#ifndef WIN32
void _splitpath(const char *path, char *drive, char *dir, char *fname, char *ext);
static void _split_whole_name(const char *whole_name, char *fname, char *ext);
#endif

int main(int argc, char** argv )
{
	if ( argc != 5 )
	{
		readme();
		return -1;
	}

	Mat img_scene, img_object;
	img_scene = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
	edge_detect(img_scene);

	if ( !img_scene.data )
	{
		std::cout << "No Scene Image \n" << std::endl;
		return -1;
	}

	int dpi = atoi( argv[4] );

	if (0 == strcmp("template", argv[3])) {
		// template match

		// 判断该dpi下的模板图片是否已经生成了
		char drive[128];
		char dir[256];
		char fname[128];
		char ext[128];
		char dst_object[256];

		_splitpath(argv[2], drive, dir, fname, ext);
		sprintf(dst_object, "./templates/%d_%s.jpg", dpi, fname);

		img_object = imread( dst_object, CV_LOAD_IMAGE_GRAYSCALE );

		if ( !img_object.data ) {
			// 还未生成
			Mat img_object_src;
			img_object_src = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );

			if ( !img_object_src.data )
			{
				std::cout << "No Object Image \n" << std::endl;
				return -1;
			}

			// 根据 dpi 放大
			resize(img_object_src, img_object, Size(int((float)img_object_src.cols * dpi / 320), int((float)img_object_src.rows * dpi / 320)));

			// 边界检测
			edge_detect(img_object);

			// 保存模板
			imwrite( dst_object, img_object );
		}

		/// Source image to display
		Mat img_display, result;
		img_scene.copyTo( img_display );

		/// Create the result matrix
		int result_cols =  img_scene.cols - img_object.cols + 1;
		int result_rows = img_scene.rows - img_object.rows + 1;

		result.create( result_rows, result_cols, CV_32FC1 );
		matchTemplate( img_scene, img_object, result, CV_TM_CCOEFF_NORMED);

		normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );

		/// Localizing the best match with minMaxLoc
		double minVal; double maxVal; Point minLoc; Point maxLoc;
		Point matchLoc;

		minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

		matchLoc = maxLoc;

		printf("%d %d", (matchLoc.x + matchLoc.x + img_object.cols) / 2, (matchLoc.y + matchLoc.y + img_object.rows) / 2);

		/// Show me what you got
		rectangle( img_display, matchLoc, Point( matchLoc.x + img_object.cols , matchLoc.y + img_object.rows ), Scalar(255, 255, 0), 4, 8, 0 );
		rectangle( result, matchLoc, Point( matchLoc.x + img_object.cols , matchLoc.y + img_object.rows ), Scalar(255, 255, 0), 4, 8, 0 );

		circle( img_display, Point((matchLoc.x + matchLoc.x + img_object.cols) / 2, (matchLoc.y + matchLoc.y + img_object.rows) / 2), 10, Scalar(255, 255, 0), 5);

		{
			//-- Save detected matches
			char dst_img[256];
			_splitpath(argv[1], drive, dir, fname, ext);
			sprintf(dst_img, "./result/template_%s.jpg", fname);
			imwrite( dst_img, img_display );
			/*
			   const char* image_window = "Source Image";
			   const char* result_window = "Result window";
			/// Create windows
			namedWindow( image_window, WINDOW_AUTOSIZE );
			namedWindow( result_window, WINDOW_AUTOSIZE );

			imshow( image_window, img_display );
			imshow( result_window, result );

			waitKey(0);
			 */
		}
	}
	else if (0 == strcmp("feature", argv[3])) {
		// feature match

		// 目标图片
		img_object = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );

		if ( !img_object.data )
		{
			std::cout << "No Object Image \n" << std::endl;
			return -1;
		}

		edge_detect(img_object);

		std::vector<KeyPoint> keypoints_object, keypoints_scene;

		Ptr<SIFT> scene_detector = SIFT::create( 10000 );
		scene_detector->detect( img_scene, keypoints_scene );

		Ptr<SIFT> object_detector = SIFT::create( 400 );
		object_detector->detect( img_object, keypoints_object );

		{
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

		//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
		std::vector< DMatch > good_matches;

		for( int i = 0; i < descriptors_object.rows; i++ )
		{
			if( matches[i].distance < 3*min_dist )
			{
				good_matches.push_back( matches[i]);
			}
		}

		if (0 >= good_matches.size()) {
			printf("have no object image in scene image");
			return -1;
		}

		Mat img_matches;
		drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
				good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

		//-- Localize the object
		std::vector<Point2f> obj;
		std::vector<Point2f> scene;

		for( unsigned int i = 0; i < good_matches.size(); i++ )
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
		printf("%f %f", center_point.x, center_point.y);

		{
			//-- Save detected matches
			char drive[128];
			char dir[256];
			char fname[128];
			char ext[128];
			char dst_img[256];

			_splitpath(argv[1], drive, dir, fname, ext);
			sprintf(dst_img, "./result/feature_%s.jpg", fname);
			imwrite( dst_img, img_matches );

			//-- Show detected matches
			/*
			   namedWindow("Good Matches & Object detection", WINDOW_NORMAL );
			   imshow( "Good Matches & Object detection", img_matches );
			   waitKey(0);
			 */
		}
	}

	return 0;
}

/** @function readme */
void readme()
{
	std::cout << "usage: sb_detect <Scene_Path> <Object_Path> <Method> <dpi>\n" << std::endl;
	std::cout << "Method : feature or template\n" << std::endl;
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

#ifndef WIN32
void _splitpath(const char *path, char *drive, char *dir, char *fname, char *ext)
{
	const char *p_whole_name;

	drive[0] = '\0';
	if (NULL == path)
	{
		dir[0] = '\0';
		fname[0] = '\0';
		ext[0] = '\0';
		return;
	}

	if ('/' == path[strlen(path)])
	{
		strcpy(dir, path);
		fname[0] = '\0';
		ext[0] = '\0';
		return;
	}

	p_whole_name = rindex(path, '/');
	if (NULL != p_whole_name)
	{
		p_whole_name++;
		_split_whole_name(p_whole_name, fname, ext);

		snprintf(dir, p_whole_name - path, "%s", path);
	}
	else
	{
		_split_whole_name(path, fname, ext);
		dir[0] = '\0';
	}
}

static void _split_whole_name(const char *whole_name, char *fname, char *ext)
{
	const char *p_ext;

	p_ext = rindex(whole_name, '.');
	if (NULL != p_ext)
	{
		strcpy(ext, p_ext);
		snprintf(fname, p_ext - whole_name + 1, "%s", whole_name);
	}
	else
	{
		ext[0] = '\0';
		strcpy(fname, whole_name);
	}
}

#endif
