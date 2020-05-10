#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include "structIO.hpp"

using namespace std;

void matchDescriptors(cv::Mat &imgSource, cv::Mat &imgRef, vector<cv::KeyPoint> &kPtsSource, vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      vector<cv::DMatch> &matches, string descriptorType, string matcherType, string selectorType, bool crossCheck, string windowName)
{
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {

        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        cout << "BF matching cross-check=" << crossCheck;
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        cout << "FLANN matching";
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::FlannBasedMatcher::create();
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        double t = (double)cv::getTickCount();
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (NN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    {
        // k nearest neighbors (k=2)

        std::vector<std::vector<cv::DMatch>> knnMatches;
        std::vector<cv::DMatch> knnMatchesFlatten;
        int k = 2;
        if (crossCheck)
            k = 1;
        double t = (double)cv::getTickCount();
        // matcher->knnMatch(descSource, descRef, matches);
        matcher->knnMatch(descSource, descRef, knnMatches, k);

        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

        if (k == 2)
        {
            float nearestNeighborDistanceRatio = 0.8;

            for (size_t i = 0; i < knnMatches.size(); i++)
            {
                if (knnMatches[i][0].distance / knnMatches[i][1].distance < nearestNeighborDistanceRatio)
                {
                    matches.push_back(knnMatches[i][0]);
                }
            }
        }
        else
        {
            for (auto &knnMatch : knnMatches)
            {
                if (knnMatch.size())

                {
                    matches.insert(std::end(matches), std::begin(knnMatch), std::end(knnMatch));
                }
            }
        }

        float discaredPecentage = 1 - (matches.size() / (float)knnMatches.size());

        cout << " (KNN) with k=" << k << " n=" << matches.size() << " matches in " << 1000 * t / 1.0
             << " ms, discared percentage = " << discaredPecentage * 100 << " %" << endl;
    }

    // visualize results
    cv::Mat matchImg = imgRef.clone();
    cv::drawMatches(imgSource, kPtsSource, imgRef, kPtsRef, matches,
                    matchImg, cv::Scalar::all(-1), cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::namedWindow(windowName, 7);
    cv::imshow(windowName, matchImg);
}

void doMatch(string kptSrcPath, string kptRefPath, string descSrcPath, string descRefPath,
             string matcherType, string descriptorType, string selectorType, string windowName, bool crossCheck)
{
    cv::Mat imgSource = cv::imread("../images/img1gray.png");
    cv::Mat imgRef = cv::imread("../images/img2gray.png");

    vector<cv::KeyPoint> kptsSource, kptsRef;

    readKeypoints(kptSrcPath.c_str(), kptsSource);
    readKeypoints(kptRefPath.c_str(), kptsRef);

    cv::Mat descSource, descRef;

    readDescriptors(descSrcPath.c_str(), descSource);
    readDescriptors(descRefPath.c_str(), descRef);

    vector<cv::DMatch> matches;

    matchDescriptors(imgSource, imgRef, kptsSource, kptsRef, descSource, descRef, matches, descriptorType, matcherType, selectorType, crossCheck, windowName);
}

void doMatch(string dataset, string matcherType, string descriptorType, string selectorType, string windowName, bool crossCheck)
{
    string kptSrcPath, kptRefPath, descSrcPath, descRefPath;

    if (dataset.compare("small") == 0)
    {
        kptSrcPath = "../dat/C35A5_KptsSource_BRISK_small.dat";
        kptRefPath = "../dat/C35A5_KptsRef_BRISK_small.dat";
        descSrcPath = "../dat/C35A5_DescSource_BRISK_small.dat";
        descRefPath = "../dat/C35A5_DescRef_BRISK_small.dat";
    }
    if (dataset.compare("large") == 0)
    {
        kptSrcPath = "../dat/C35A5_KptsSource_BRISK_large.dat";
        kptRefPath = "../dat/C35A5_KptsRef_BRISK_large.dat";
        descSrcPath = "../dat/C35A5_DescSource_BRISK_large.dat";
        descRefPath = "../dat/C35A5_DescRef_BRISK_large.dat";
    }
    if (dataset.compare("SIFT") == 0)
    {
        kptSrcPath = "../dat/C35A5_KptsSource_SIFT.dat";
        kptRefPath = "../dat/C35A5_KptsRef_SIFT.dat";
        descSrcPath = "../dat/C35A5_DescSource_SIFT.dat";
        descRefPath = "../dat/C35A5_DescRef_SIFT.dat";
    }

    doMatch(kptSrcPath, kptRefPath, descSrcPath, descRefPath, matcherType, descriptorType, selectorType, windowName, crossCheck);
}

int main()
{

    std::cout << "---------------------------------------------- MAT_BF ----------------------------------------------" << endl;

    std::cout << "----------------------- BRISK small -----------------------" << endl;

    doMatch("small", "MAT_BF", "DES_BINARY", "SEL_NN",
            "BRISK small - NN: Matching keypoints between two camera images crossCheck: OFF", false);

    doMatch("small", "MAT_BF", "DES_BINARY", "SEL_NN",
            "BRISK small - NN: Matching keypoints between two camera images crossCheck: OFF", true);

    doMatch("small", "MAT_BF", "DES_BINARY", "SEL_KNN",
            "BRISK small - KNN: Matching keypoints between two camera images crossCheck: OFF", false);

    doMatch("small", "MAT_BF", "DES_BINARY", "SEL_KNN",
            "BRISK small - KNN: Matching keypoints between two camera images crossCheck: OFF", true);

    std::cout << "*********************** BRISK small ***********************" << endl;

    std::cout << "----------------------- BRISK large -----------------------" << endl;

    doMatch("large", "MAT_BF", "DES_BINARY", "SEL_NN",
            "BRISK large - NN: Matching keypoints between two camera images crossCheck: OFF", false);

    doMatch("large", "MAT_BF", "DES_BINARY", "SEL_NN",
            "BRISK large - NN: Matching keypoints between two camera images crossCheck: OFF", true);

    doMatch("large", "MAT_BF", "DES_BINARY", "SEL_KNN",
            "BRISK large - KNN: Matching keypoints between two camera images crossCheck: OFF", false);

    doMatch("large", "MAT_BF", "DES_BINARY", "SEL_KNN",
            "BRISK large - KNN: Matching keypoints between two camera images crossCheck: OFF", true);

    std::cout << "*********************** BRISK large ***********************" << endl;

    std::cout << "----------------------- SIFT -----------------------" << endl;

    doMatch("SIFT", "MAT_BF", "NORM_L2", "SEL_NN",
            "SIFT - NN: Matching keypoints between two camera images crossCheck: OFF", false);

    doMatch("SIFT", "MAT_BF", "NORM_L2", "SEL_NN",
            "SIFT - NN: Matching keypoints between two camera images crossCheck: OFF", true);

    doMatch("SIFT", "MAT_BF", "NORM_L2", "SEL_KNN",
            "SIFT - KNN: Matching keypoints between two camera images crossCheck: OFF", false);

    doMatch("SIFT", "MAT_BF", "NORM_L2", "SEL_KNN",
            "SIFT - KNN: Matching keypoints between two camera images crossCheck: OFF", true);

    std::cout << "*********************** SIFT ***********************" << endl;

    std::cout << "*********************************************** MAT_BF **********************************************" << endl;

    std::cout << "---------------------------------------------- MAT_FLANN ----------------------------------------------" << endl;

    std::cout << "----------------------- BRISK large -----------------------" << endl;

    doMatch("large", "MAT_FLANN", "DES_BINARY", "SEL_NN",
            "FLANN large - NN: Matching keypoints between two camera images crossCheck: OFF", false);

    doMatch("large", "MAT_FLANN", "DES_BINARY", "SEL_NN",
            "FLANN large - NN: Matching keypoints between two camera images crossCheck: OFF", true);

    doMatch("large", "MAT_FLANN", "DES_BINARY", "SEL_KNN",
            "FLANN large - KNN: Matching keypoints between two camera images crossCheck: OFF", false);

    doMatch("large", "MAT_FLANN", "DES_BINARY", "SEL_KNN",
            "FLANN large - KNN: Matching keypoints between two camera images crossCheck: OFF", true);

    std::cout << "*********************** BRISK large ***********************" << endl;

    std::cout << "----------------------- SIFT -----------------------" << endl;

    doMatch("SIFT", "MAT_FLANN", "DES_BINARY", "SEL_NN",
            "SIFT - NN: Matching keypoints between two camera images crossCheck: OFF",
            false /* can also be true, this value is not used anyways*/);

    doMatch("SIFT", "MAT_FLANN", "DES_BINARY", "SEL_KNN",
            "SIFT - KNN: Matching keypoints between two camera images crossCheck: OFF", false);

    doMatch("SIFT", "MAT_FLANN", "DES_BINARY", "SEL_KNN",
            "SIFT - KNN: Matching keypoints between two camera images crossCheck: OFF", true);

    std::cout << "*********************** SIFT ***********************" << endl;

    std::cout << "*********************************************** MAT_FLANN **********************************************" << endl;

    cv::waitKey(0);
}