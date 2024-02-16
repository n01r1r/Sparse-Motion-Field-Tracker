#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>


#ifdef _DEBUG 
#pragma comment (lib, "opencv_world480d.lib") 
#else
#pragma comment (lib, "opencv_world480.lib")
#endif


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

    Mat old_frame, old_gray;
    vector<Point2f> p0, p1;
    vector<float> speeds;

    // take the first frame and find features
    capture >> old_frame;
    cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);

    // draw ROI
    Rect2d roi = selectROI(old_frame);
    waitKey(0);
    Mat roi_gray = old_gray(roi);

    goodFeaturesToTrack(roi_gray, p0, 300, 0.05, 5, Mat(), 7, true, 0.04);

    for (auto i = 0; i < p0.size(); i++) {
        p0[i].x += roi.x;
        p0[i].y += roi.y;
    }

    // init mask for drawing
    Mat mask = Mat::zeros(old_frame.size(), old_frame.type());
    Mat overlay = imread(imagename);
    Mat frame, frame_gray;


    // frame loop
    bool loop = false;
    bool useMask = false;

    while (!loop) {
        capture >> frame;
        if (frame.empty()) break;
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

        // calculate optical flow
        vector<uchar> status;
        vector<float> err;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15, 15), 2, criteria);


        vector<Point2f> good_new;

        Mat ransac;
        vector<uchar> inliers;
        if (p0.size() > 4 || p1.size() > 4) {
            ransac = findHomography(p0, p1, RANSAC, 3, inliers);

        }
        else {
            cout << "\n\n NOT ENOUGH POINTS TO TRACK\n\n";
            break;
        }

        for (uint i = 0; i < p0.size(); i++) {
            // select good points
            if (status[i] == (1) && inliers[i] == 1) {
                good_new.push_back(p1[i]);
                // draw the tracks
                //line(mask, p1[i], p0[i], colors[i], 2);
                circle(frame, p1[i], 2, Scalar(0, 255, 0), -1);
            }
        }

        if (useMask) {
            RotatedRect newROI = minAreaRect(good_new);
            Point2f pts[4];
            newROI.points(pts);
            //
            sortCorners(pts);
            //
            vector<Point2f> overlayCorners = { Point2f(0, 0), Point2f(overlay.cols, 0), Point2f(overlay.cols, overlay.rows), Point2f(0, overlay.rows) };
            vector<Point2f> newroiCorners = { pts[0], pts[1], pts[2], pts[3] };
            Mat H = findHomography(overlayCorners, newroiCorners);
            Mat warpedOverlay;
            cv::warpPerspective(overlay, warpedOverlay, H, frame.size());
            frame += warpedOverlay;
        }

        // ROI check
        if (good_new.empty() || p0.empty() || p1.empty()) {
            cout << "\n\n ROI IS OUT OF THE IMAGE \n\n";
            loop = true;
        }
        else {
            p0 = good_new;
        }

        Mat img;
        cv::add(frame, mask, img);

        int keyboard = waitKey(25);
        if (keyboard == 'q' || keyboard == 27) break;
        old_gray = frame_gray.clone();

        cvui::window(img, 10, 50, 120, 150, "Settings");
        cvui::checkbox(img, 15, 80, "TERMINATE?", &loop);
        cvui::checkbox(img, 15, 100, "MASK", &useMask);
        cvui::text(img, 15, 120, to_string(p0.size()));
        cvui::update();
        cv::imshow(WINDOW_NAME, img);
    }

    return 0;
}


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
