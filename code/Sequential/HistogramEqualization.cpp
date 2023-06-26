#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/utility.hpp>
#include <chrono>
using namespace cv;



void histogramEqualization(Mat& image, Mat& yuv, int num_rows, int num_cols) {    
    int total_pixels = num_rows * num_cols;
    std::vector<int> histogram(256, 0);   
    
    // Perform the color conversion and calculate the histogram

    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            Vec3b intensity = image.at<Vec3b>(i, j);

            uchar B = intensity.val[0];
            uchar G = intensity.val[1];
            uchar R = intensity.val[2];

            uchar Y = 0.299 * R + 0.587 * G + 0.114 * B;
            uchar U = -0.169 * R - 0.331 * G + 0.499 * B + 128;
            uchar V = 0.499 * R - 0.418 * G - 0.0813 * B + 128;

            ++histogram[Y];

            Vec3b yuv_int = yuv.at<Vec3b>(i, j);
            yuv_int[0] = Y;
            yuv_int[1] = U;
            yuv_int[2] = V;

            yuv.at<Vec3b>(i, j) = yuv_int;
        }
    }
    
    //  Calculate the cumulative distribution function (CDF) and the
    // and the index of the min element of CDF greather than zero
    std::vector<int> cdf(256, 0);
    cdf[0] = histogram[0];
    int cdf_min = 0;
    for (int i = 1; i < 256; ++i) {
        cdf[i] = cdf[i - 1] + histogram[i];
        
    }
    for (int i = 0; i < 256; ++i) {
        if (cdf[i] != 0) {
            cdf_min = i;
            break;
        }
    }
    //std::cout << cdf_min << std::endl;

    // Perform histogram equalization
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            Vec3b yuv_int = yuv.at<Vec3b>(i, j);

            uchar Y = static_cast<int>(round(255 * (cdf[yuv_int[0]] - cdf_min) / static_cast<float>(total_pixels - cdf_min)));
            uchar U = yuv_int[1];
            uchar V = yuv_int[2];

            uchar R = std::max(0, std::min(255, static_cast<int>(Y + 1.402 * (V - 128))));
            uchar G = std::max(0, std::min(255, static_cast<int>(Y - 0.344 * (U - 128) - 0.714 * (V - 128))));
            uchar B = std::max(0, std::min(255, static_cast<int>(Y + 1.772 * (U - 128))));

            Vec3b intensity = yuv.at<Vec3b>(i, j);
            intensity[0] = B;
            intensity[1] = G;
            intensity[2] = R;
            yuv.at<Vec3b>(i, j) = intensity;

            
        }
    }
}
int main() {
    //Load and show the image
    std::string image_path = samples::findFile("mountain.jpg");
    Mat image = imread(image_path);
    resize(image, image, Size(2000, 2000), INTER_NEAREST);
    std::vector<int> times(10, 0);
    int num_rows = image.size[0];
    int num_cols = image.size[1];
    Mat yuv(Size(num_cols, num_rows), CV_8UC3);

    for (int i = 0; i < 10; ++i) {
        //imshow("Display window 1", image);
        //waitKey(0);

        // Perform histogram equalization
        auto start = std::chrono::system_clock::now();
        histogramEqualization(image, yuv, num_rows, num_cols);
        std::chrono::duration<double> duration = (std::chrono::system_clock::now() - start);

        //std::cout << "Finished in:" << duration.count()*1000<< " milliseconds" << std::endl;
        times[i] = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        }
    // Show and save the image
    float mean_time = 0;
    for (int i = 0; i < 10; ++i) {
        mean_time += times[i];
    }

    std::cout << "Average execution time : " << mean_time/times.size() << " milliseconds" << std::endl; 
    //imshow("Display window 2", yuv);
    //waitKey(0);
    imwrite("test2.png", yuv);
    return 0;
}

