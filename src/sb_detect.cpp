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

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
    	std::cout << "usage: " << argv[0] << " <Image_Path>\n" << std::endl;
        return -1;
    }

    Mat img_scene, img_object;
    img_scene = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
    img_object = imread( "/home/wuxingyu/sb_helper/qp_logo_03.png", CV_LOAD_IMAGE_GRAYSCALE );
//    img_object = imread( "/home/wuxingyu/sb_helper/qz_logo_07.png", CV_LOAD_IMAGE_GRAYSCALE );
//    img_object = imread( "/home/wuxingyu/opencv/samples/data/box.png", CV_LOAD_IMAGE_GRAYSCALE );

    if ( !img_scene.data )
    {
    	std::cout << "No image data \n" << std::endl;
        return -1;
    }

    // laplace edge detect
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    Mat abs_dst, dst;

    /// Remove noise by blurring with a Gaussian filter
    GaussianBlur( img_scene, img_scene, Size(3,3), 0, 0, BORDER_DEFAULT );

    /// Apply Laplace function
    Laplacian( img_scene, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( dst, img_scene );

    GaussianBlur( img_object, img_object, Size(3,3), 0, 0, BORDER_DEFAULT );

    /// Apply Laplace function
    Laplacian( img_object, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( dst, img_object );

    std::vector<KeyPoint> keypoints_object, keypoints_scene;

    Ptr<SIFT> scene_detector = SIFT::create( 10000 );
    scene_detector->detect( img_scene, keypoints_scene );

    Ptr<SIFT> object_detector = SIFT::create( 400 );
    object_detector->detect( img_object, keypoints_object );

    Mat img_keypoints_scene, img_keypoints_object;
        drawKeypoints( img_object, keypoints_object, img_keypoints_object, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
        drawKeypoints( img_scene, keypoints_scene, img_keypoints_scene, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

        namedWindow("Display Image", WINDOW_AUTOSIZE );
        imshow("Display Image", img_keypoints_scene);

        namedWindow("Target Image", WINDOW_AUTOSIZE );
        imshow("Target Image", img_keypoints_object);

/*
        waitKey(0);

        return 0;
*/
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

    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );

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

    printf("the size of good_matches is %d\n", good_matches.size());
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

    //-- Show detected matches
    imshow( "Good Matches & Object detection", img_matches );
/*
    Mat img_keypoints_1, img_keypoints_2;
    drawKeypoints( image, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    drawKeypoints( image_target, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", img_keypoints_1);

    namedWindow("Target Image", WINDOW_AUTOSIZE );
    imshow("Target Image", img_keypoints_2);
*/
//    imwrite( "./img_keypoint.jpg", img_keypoints_1 );
/*
    Mat gray_image;
    cvtColor( image, gray_image, CV_BGR2GRAY );
    namedWindow("Gray Image", WINDOW_AUTOSIZE );
    imshow("Gray Image", gray_image);
*/
    waitKey(0);

    return 0;
}



