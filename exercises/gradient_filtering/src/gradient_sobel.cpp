#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

void gradientSobel()
{
    // TODO: Based on the image gradients in both x and y, compute an image
    // which contains the gradient magnitude according to the equation at the
    // beginning of this section for every pixel position. Also, apply different
    // levels of Gaussian blurring before applying the Sobel operator and compare the results.

    // load image from file
    cv::Mat img;
    img = cv::imread("../images/img1.png");

    // convert image to grayscale
    cv::Mat imgGray;
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    // show result
    string windowName = "Intput";
    cv::namedWindow(windowName, 1); // create window
    cv::imshow(windowName, img);

    // create filter kernel
    float gauss_data[25] = {1, 4, 7, 4, 1,
                            4, 16, 26, 16, 4,
                            7, 26, 41, 26, 7,
                            4, 16, 26, 16, 4,
                            1, 4, 7, 4, 1};
    cv::Mat gauss_kernel = (cv::Mat(5, 5, CV_32F, gauss_data)) / 273.0;

    // apply filter
    cv::Mat result;
    // cv::filter2D(img, img, -1, gauss_kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    cv::GaussianBlur(imgGray, imgGray, cv::Size(5, 5), 2.0);

    // show result
    windowName = "Gaussian output";
    cv::namedWindow(windowName, 1); // create window
    cv::imshow(windowName, imgGray);

    // create filter kernel
    float sobel_x[9] = {-1, 0, +1,
                        -2, 0, +2,
                        -1, 0, +1};

    float sobel_y[9] = {-1, -2, -1,
                        -2, 0, +2,
                        +1, +2, +1};
    cv::Mat kernel_x = cv::Mat(3, 3, CV_32F, sobel_x);
    cv::Mat kernel_y = cv::Mat(3, 3, CV_32F, sobel_y);

    // apply filter
    cv::Mat result_x;
    cv::Mat result_y;
    cv::Mat result_xy;

    cv::filter2D(imgGray, result_x, -1, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(imgGray, result_y, -1, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    // show result
    // windowName = "Sobel operator (x-direction)";
    // cv::namedWindow(windowName, 1); // create window
    // cv::imshow(windowName, result_x);

    // windowName = "Sobel operator (y-direction)";
    // cv::namedWindow(windowName, 1); // create window
    // cv::imshow(windowName, result_y);

    cv::Mat magnitude = imgGray.clone();
    for (int r = 0; r < magnitude.rows; r++)
    {
        for (int c = 0; c < magnitude.cols; c++)
        {
            magnitude.at<unsigned char>(r, c) = sqrt(pow(result_x.at<unsigned char>(r, c), 2) +
                                                     pow(result_y.at<unsigned char>(r, c), 2));
        }
    }

    windowName = "Intensity magnitude";
    cv::namedWindow(windowName, 1); // create window
    cv::imshow(windowName, magnitude);
    cv::waitKey(0); // wait for keyboard input before continuing
}

int main()
{
    gradientSobel();
}