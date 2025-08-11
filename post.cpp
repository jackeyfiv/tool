#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

cv::Mat keep_largest_region(const cv::Mat& mask) {
    // 如果mask全0，直接返回
    if (cv::countNonZero(mask) == 0) {
        return mask.clone();
    }

    // 连通域分析
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);

    // 如果没有找到连通区域，返回原图
    if (num_labels <= 1) {
        return mask.clone();
    }

    // 找到最大区域（跳过背景区域0）
    int max_area = 0;
    int max_label = 1;
    for (int i = 1; i < num_labels; ++i) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) > max_area) {
            max_area = stats.at<int>(i, cv::CC_STAT_AREA);
            max_label = i;
        }
    }

    // 创建结果mask
    cv::Mat result_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::compare(labels, max_label, result_mask, cv::CMP_EQ);
    
    return result_mask;
}

cv::Mat image_fill(const cv::Mat& image) {
    cv::Mat src;
    // 确保是二值图像
    if (image.channels() > 1) {
        cv::cvtColor(image, src, cv::COLOR_BGR2GRAY);
    } else {
        src = image.clone();
    }
    
    // 二值化处理（如果还不是二值）
    cv::Mat ori;
    if (src.type() != CV_8UC1) {
        src.convertTo(ori, CV_8UC1);
    } else {
        ori = src.clone();
    }
    
    cv::threshold(ori, ori, 127, 255, cv::THRESH_BINARY);
    src = ori.clone();

    // 创建填充掩码（比原图大一圈）
    cv::Mat mask = cv::Mat::zeros(ori.rows + 2, ori.cols + 2, CV_8UC1);
    
    // 计算中心点
    cv::Point center(ori.cols / 2, ori.rows / 2);
    
    // 漫水填充
    cv::floodFill(
        src, 
        mask, 
        center, 
        255, 
        nullptr, 
        cv::Scalar(), 
        cv::Scalar(), 
        cv::FLOODFILL_FIXED_RANGE | (255 << 8)
    );
    
    // 计算填充结果
    cv::Mat output = ori - src;
    
    // 创建圆形掩码
    cv::Mat circle_mask = cv::Mat::zeros(ori.size(), CV_8UC1);
    int radius = std::min(center.x, center.y) - 5;
    if (radius > 0) {
        cv::circle(circle_mask, center, radius, 255, -1);
    }
    
    // 应用圆形掩码
    cv::bitwise_and(output, circle_mask, output);
    
    return output;
}

// 使用示例
int main() {
    // 读取输入图像（假设是二值分割结果）
    cv::Mat input = cv::imread("segmentation.png", cv::IMREAD_GRAYSCALE);
    
    if (input.empty()) {
        std::cerr << "Error: Could not read image" << std::endl;
        return -1;
    }
    
    // 处理1：保留最大连通区域
    cv::Mat largest_region = keep_largest_region(input);
    
    // 处理2：漫水填充处理
    cv::Mat filled_result = image_fill(largest_region);
    
    // 显示结果
    cv::imshow("Original", input);
    cv::imshow("Largest Region", largest_region);
    cv::imshow("Filled Result", filled_result);
    cv::waitKey(0);
    
    // 保存结果
    cv::imwrite("largest_region.png", largest_region);
    cv::imwrite("filled_result.png", filled_result);
    
    return 0;
}
