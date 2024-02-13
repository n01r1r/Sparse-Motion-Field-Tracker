/*
* Created at 2024-02-13
* Created by Dongyeob Han (n01r1r @github)
* Masking ROI region with sparse motion field tracking
* OpenCV 4.8.0
* CVUI 2.7.0
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>  //findHomography()
#include <opencv2/highgui.hpp>  //selectROI()
#include <opencv2/imgproc.hpp>  //warpPerspective()
#include <opencv2/video.hpp>    //calcOpticalFlowPyrLK()

#pragma comment (lib, "opencv_world480d.lib")

using namespace cv;
using namespace std;

#define CVUI_IMPLEMENTATION
#define WINDOW_NAME "RESULTS"
#include "cvui.h"

void sortCorners(Point2f pts[4]);

int main()
{
    // video read
    const string& videoname = "staple.mp4";
    const string& imagename = "hmm.webp";
    VideoCapture capture(videoname);
    if (!capture.isOpened()) {
        cerr << "\n\nUnable to open file!\n\n";
        return 0;
    }

    // cvui init
    namedWindow(WINDOW_NAME);
    cvui::init(WINDOW_NAME);

    // init variables
    bool loop = false;
    bool useMask = false;
    Mat old_frame, old_gray;
    Mat mask = Mat::zeros(old_frame.size(), old_frame.type());
    Mat overlay = imread("hmm.webp");
    Mat frame, frame_gray, H, warpedOverlay;
    vector<Point2f> p0, p1, overlayCorners, newroiCorners;


    capture >> old_frame;
    cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);

    // Set ROI
    Rect2d roi = selectROI(old_frame);
    waitKey(0);
    Mat roi_gray = old_gray(roi);
    goodFeaturesToTrack(roi_gray, p0, 1500, 0.02, 5, Mat(), 7, true, 0.04);

    for (auto i = 0; i < p0.size(); i++) {
        p0[i].x += roi.x;
        p0[i].y += roi.y;
    }
    
    while (!loop) {
        capture >> frame;
        if (frame.empty()) break;
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

        // calculate optical flow for already-found feature points
        vector<uchar> status;
        vector<float> err;
        vector<Point2f> good_new;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 10, 0.5);
        calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15, 15), 2, criteria);


        for (uint i = 0; i < p0.size(); i++) {
            if (status[i] == (1)) {
                good_new.push_back(p1[i]);
                //line(mask, p1[i], p0[i], colors[i], 2);       // for tracked line drawing
                circle(frame, p1[i], 2, Scalar(0, 255, 0), -1); // for feature point dots
            }
        }
        
        RotatedRect newROI = minAreaRect(good_new);
        Point2f pts[4];
        newROI.points(pts);
        sortCorners(pts);

        bool init = true;
        if (init) {
            overlayCorners = { Point2f(0, 0), Point2f(overlay.cols, 0), Point2f(overlay.cols, overlay.rows), Point2f(0, overlay.rows) };
            newroiCorners  = { pts[0], pts[1], pts[2], pts[3] };
            !init;
        }
        else {
            newroiCorners  = { pts[0], pts[1], pts[2], pts[3] };
        }
        
        H = findHomography(overlayCorners, newroiCorners, RANSAC);
        Mat warpedOverlay;
        warpPerspective(overlay, warpedOverlay, H, frame.size());
        if(useMask) frame += warpedOverlay;
        overlayCorners = newroiCorners; // 순차적으로 homography를 계산하기 위해 corner point 갱신
        
        // ROI boundary check
        if (good_new.empty() || p0.empty() || p1.empty()) {
            cout << "\n\n ROI IS OUT OF THE IMAGE \n\n";
            loop = true;
        }
        else {
            p0 = good_new; // 순차적으로 Optical Flow를 계산하기 위해 feature point 갱신
        }
        old_gray = frame_gray.clone(); // frame update

        // cvui update
        cvui::window(frame, 10, 50, 120, 150, "Settings");
        cvui::checkbox(frame, 15, 80, "TERMINATE?", &loop);
        cvui::checkbox(frame, 15, 100, "USE MASK", &useMask);
        cvui::text(frame, 15, 120, to_string(p0.size()));
        cvui::update();
        cv::imshow(WINDOW_NAME, frame);
    }

    return 0;
}

/*
* RotatedRect로 정의되는 ROI 영역의 각 corner에 대해서
* corner 순서가 일관되게 유지될 수 있도록 정렬해줌
*/
void sortCorners(Point2f pts[4]) {
    Point2f center(0.f, 0.f);

    // compute center;
    for (int i = 0; i < 4; i++) {
        center += pts[i];
    }
    center /= 4.f;

    sort(pts, pts + 4, [center](Point2f a, Point2f b) {
        return atan2(a.y - center.y, a.x - center.x) < atan2(b.y - center.y, b.x - center.x);
        });
}
