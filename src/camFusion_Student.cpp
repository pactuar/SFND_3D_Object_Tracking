
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <set>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 2);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // loop over matches between previous and current frame
    for (auto itMatch = kptMatches.begin(); itMatch != kptMatches.end(); ++itMatch)
    {
        // pull out identifiers for the current and previous keypoints from this match
        int prevKeyIdx = itMatch->queryIdx;
        int currKeyIdx = itMatch->trainIdx;

        // get this match's corresponding current frame keypoint and previous frame keypoint
        cv::KeyPoint prevKeypoint = kptsPrev[prevKeyIdx];
        cv::KeyPoint currKeypoint = kptsCurr[currKeyIdx];

        // check if this bounding box has the current and previous keypoints from this match
        if (boundingBox.roi.contains(currKeypoint.pt) && boundingBox.roi.contains(prevKeypoint.pt))
        {
            boundingBox.keypoints.push_back(currKeypoint);
            boundingBox.kptMatches.push_back(*itMatch);
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // STUDENT TASK (replacement for meanDistRatio)
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
    // EOF STUDENT TASK
}

// Custom comparator for the x component of lidar points
bool LidarComparatorX(LidarPoint a, LidarPoint b)
{
    return a.x < b.x;
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // auxiliary variables
    double dT = 1.0 / (frameRate + 1e-8);        // time between two measurements in seconds
    double laneWidth = 4.0;                      // assumed width of the ego lane

    // compute the median of the previous and current lidar points
    // how to sort vector: https://en.cppreference.com/w/cpp/algorithm/sort
    std::sort(lidarPointsPrev.begin(), lidarPointsPrev.end(), LidarComparatorX);
    std::sort(lidarPointsCurr.begin(), lidarPointsCurr.end(), LidarComparatorX);

    long medIndexPrev = floor(lidarPointsPrev.size() / 2.0);
    double medXPrev = lidarPointsPrev.size() % 2 == 0 ? (lidarPointsPrev[medIndexPrev - 1].x + lidarPointsPrev[medIndexPrev].x) / 2.0 : lidarPointsPrev[medIndexPrev].x;

    long medIndexCurr = floor(lidarPointsCurr.size() / 2.0);
    double medXCurr = lidarPointsCurr.size() % 2 == 0 ? (lidarPointsCurr[medIndexCurr - 1].x + lidarPointsCurr[medIndexCurr].x) / 2.0 : lidarPointsCurr[medIndexCurr].x;

    // // find closest distance to Lidar points within ego lane
    // double minXPrev = 1e9, minXCurr = 1e9;
    // for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    // {
    //     if ((it->y >= -laneWidth/2.0) && (it->y <= laneWidth/2.0))
    //     {
    //         minXPrev = minXPrev > it->x ? it->x : minXPrev;
    //     }
    // }

    // for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    // {
    //     if ((it->y >= -laneWidth/2.0) && (it->y <= laneWidth/2.0))
    //     {
    //         minXCurr = minXCurr > it->x ? it->x : minXCurr;
    //     }
    // }

    // compute TTC from both measurements
    TTC = medXCurr * dT / (medXPrev - medXCurr);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    std::multimap<int, int> bbCandidateMatches;

    // loop over matches between previous and current frame
    for (auto itMatch = matches.begin(); itMatch != matches.end(); ++itMatch)
    {
        // pull out identifiers for the current and previous keypoints from this match
        int prevKeyIdx = itMatch->queryIdx;
        int currKeyIdx = itMatch->trainIdx;

        // get this match's corresponding current frame keypoint and previous frame keypoint
        cv::KeyPoint prevKeypoint = prevFrame.keypoints[prevKeyIdx];
        cv::KeyPoint currKeypoint = currFrame.keypoints[currKeyIdx];
        
        // loop over all the current frame's bounding boxes
        for (auto itCurrBB = currFrame.boundingBoxes.begin(); itCurrBB != currFrame.boundingBoxes.end(); ++itCurrBB)
        {
            // check if this bounding box has the current frame keypoint
            if (itCurrBB->roi.contains(currKeypoint.pt))
            {
                // loop and look for a previous frame bounding box that contains the prev frame keypoint
                for (auto itPrevBB = prevFrame.boundingBoxes.begin(); itPrevBB != prevFrame.boundingBoxes.end(); ++itPrevBB)
                {
                    if (itPrevBB->roi.contains(prevKeypoint.pt))
                    {
                        // add the bounding box for the current frame and the previous frame to a multimap (https://www.geeksforgeeks.org/multimap-associative-containers-the-c-standard-template-library-stl/)
                        // (previousID, currentID)
                        bbCandidateMatches.insert(pair<int, int>(itPrevBB->boxID, itCurrBB->boxID));
                    }
                }
            }
        }
    }

    // loop over candidates and count how many times it has a specific pair
    // loop over all the current frame's bounding boxes
    std::vector<std::tuple<int, int, int>> bbMatchCounts;  //https://en.cppreference.com/w/cpp/utility/tuple
    for (auto itCurrBB = currFrame.boundingBoxes.begin(); itCurrBB != currFrame.boundingBoxes.end(); ++itCurrBB)
    {
        for (auto itPrevBB = prevFrame.boundingBoxes.begin(); itPrevBB != prevFrame.boundingBoxes.end(); ++itPrevBB)
        {
            int count = 0;
            for (auto itCandidatePair = bbCandidateMatches.begin(); itCandidatePair != bbCandidateMatches.end(); ++itCandidatePair)
            {
                if ((itCandidatePair->first == itPrevBB->boxID) && (itCandidatePair->second == itCurrBB->boxID))
                {
                    count++;
                }
            }
            bbMatchCounts.push_back(std::make_tuple(itPrevBB->boxID, itCurrBB->boxID, count));
        }
    }

    // create set of previous bb ids and current bb ids: https://www.geeksforgeeks.org/set-in-cpp-stl/
    std::set<int> prevBB_ids;
    std::set<int> currBB_ids;

    for (auto itCurrBB = currFrame.boundingBoxes.begin(); itCurrBB != currFrame.boundingBoxes.end(); ++itCurrBB)
    {
        currBB_ids.insert(itCurrBB->boxID);
    }

    for (auto itPrevBB = prevFrame.boundingBoxes.begin(); itPrevBB != prevFrame.boundingBoxes.end(); ++itPrevBB)
    {
        prevBB_ids.insert(itPrevBB->boxID);
    }

    // see which set has less bounding boxes (this can be the max number of matches ... assuming exclusive matching)
    bool prevImageHasLessBBs = prevBB_ids.size() < currBB_ids.size() ? true : false;

    if (prevImageHasLessBBs)
    {
        // loop over the previous BBs
        for (auto itPrevBB = prevFrame.boundingBoxes.begin(); itPrevBB != prevFrame.boundingBoxes.end(); ++itPrevBB)
        {
                int maxMatches = 0;
                int bestCurrId = 0;
                
                // loop over the match counts
                for (auto itMatchCount = bbMatchCounts.begin(); itMatchCount != bbMatchCounts.end(); ++itMatchCount)
                {
                    if (std::get<0>(*itMatchCount) == itPrevBB->boxID)
                    {
                        if (std::get<2>(*itMatchCount) > maxMatches && (currBB_ids.count(std::get<1>(*itMatchCount)) == 1))
                        {
                            maxMatches = std::get<2>(*itMatchCount);
                            bestCurrId = std::get<1>(*itMatchCount);
                        }
                    }
                }

                // Take the best, but also remove it from the set so it can't be used again
                bbBestMatches.insert(pair<int, int>(itPrevBB->boxID, bestCurrId));
        }
    }
    else
    {
        // loop over the current BBs
        for (auto itCurrBB = currFrame.boundingBoxes.begin(); itCurrBB != currFrame.boundingBoxes.end(); ++itCurrBB)
        {
                int maxMatches = 0;
                int bestPrevId = 0;
                
                // loop over the match counts
                for (auto itMatchCount = bbMatchCounts.begin(); itMatchCount != bbMatchCounts.end(); ++itMatchCount)
                {
                    if (std::get<1>(*itMatchCount) == itCurrBB->boxID)
                    {
                        if (std::get<2>(*itMatchCount) > maxMatches && (prevBB_ids.count(std::get<0>(*itMatchCount)) == 1))
                        {
                            maxMatches = std::get<2>(*itMatchCount);
                            bestPrevId = std::get<0>(*itMatchCount);
                        }
                    }
                }

                // Take the best, but also remove it from the set so it can't be used again
                bbBestMatches.insert(pair<int, int>(bestPrevId, itCurrBB->boxID));
        }
    }


    // std::cout << "Match Candidates" << std::endl;
    // for (auto it = bbCandidateMatches.begin(); it != bbCandidateMatches.end(); ++it){
    //     cout << '\t' << it->first << '\t' << it->second << std::endl;
    // }

    std::cout << "Match Counts" << std::endl;
    for (auto it = bbMatchCounts.begin(); it != bbMatchCounts.end(); ++it)
    {
        cout << '\t' << std::get<0>(*it) << '\t' << std::get<1>(*it) << '\t' << std::get<2>(*it) << std::endl;
    }

    std::cout << "Best Matches" << std::endl;
    for (auto it = bbBestMatches.begin(); it != bbBestMatches.end(); ++it)
    {
        cout << '\t' << it->first << '\t' << it->second << std::endl;
    }

    std::cout << "Matching Complete" << endl;

}
