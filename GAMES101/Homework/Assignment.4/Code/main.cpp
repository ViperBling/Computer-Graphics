/*
 * @Author: Zero
 * @Date: 2020-12-15 18:26:22
 * @LastEditTime: 2020-12-16 16:51:13
 * @Description: 
 * @可以输入预定的版权声明、个性签名、空行等
 */
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

std::vector<cv::Point2f> control_points;

void mouse_handler(int event, int x, int y, int flags, void *userdata) 
{
    if (event == cv::EVENT_LBUTTONDOWN && control_points.size() < 6) 
    {
        std::cout << "Left button of the mouse is clicked - position (" << x << ", "
        << y << ")" << '\n';
        control_points.emplace_back(x, y);
    }     
}

void naive_bezier(const std::vector<cv::Point2f> &points, cv::Mat &window) 
{
    auto &p_0 = points[0];
    auto &p_1 = points[1];
    auto &p_2 = points[2];
    auto &p_3 = points[3];

    for (double t = 0.0; t <= 1.0; t += 0.001) 
    {
        auto point = std::pow(1 - t, 3) * p_0 + 3 * t * std::pow(1 - t, 2) * p_1 +
                 3 * std::pow(t, 2) * (1 - t) * p_2 + std::pow(t, 3) * p_3;

        window.at<cv::Vec3b>(point.y, point.x)[2] = 255;
    }
}

cv::Point2f recursive_bezier(const std::vector<cv::Point2f> &control_points, float t) 
{
    // TODO: Implement de Casteljau's algorithm
    std::vector<cv::Point2f> vec;
    if (control_points.size() == 2)  
        return (1-t) * control_points[0] + t * control_points[1];
    
    for (int i = 0; i < control_points.size() - 1; ++i)
        vec.emplace_back((1-t) * control_points[i] + t * control_points[i+1]);
    
    return recursive_bezier(vec, t);
}

void bezier(const std::vector<cv::Point2f> &control_points, cv::Mat &window) 
{
    // TODO: Iterate through all t = 0 to t = 1 with small steps, and call de Casteljau's 
    // recursive Bezier algorithm.
    for (double t = 0.0; t <= 1.0; t += 0.001) {
        auto point = recursive_bezier(control_points, t);
        // window.at<cv::Vec3b>(point.y, point.x)[0] = 255;         // B
        window.at<cv::Vec3b>(point.y, point.x)[1] = 255;            // G
        // window.at<cv::Vec3b>(point.y, point.x)[2] = 255;         // R


        // anti aliasing
        float x = point.x - std::floor(point.x);
        float y = point.y - std::floor(point.y);
        int x_flag = x < 0.5f ? -1 : 1;
        int y_flag = y < 0.5f ? -1 : 1;

        // 当前位置的临近4个像素，p00是像素中心点
        auto p00 = cv::Point2f(std::floor(point.x) + 0.5f, std::floor(point.y) + 0.5f);
        // 左边（假设point的位置在当前像素左下的位置，那么离它最近的4个像素点位置在它左侧包围）
        auto p01 = cv::Point2f(std::floor(point.x + x_flag * 1.0f) + 0.5f, std::floor(point.y) + 0.5f);
        // 下边
        auto p10 = cv::Point2f(std::floor(point.x) + 0.5f, std::floor(point.y + y_flag * 1.0f) + 0.5f);
        // 左下
        auto p11 = cv::Point2f(std::floor(point.x + x_flag * 1.0f) + 0.5f, std::floor(point.y + y_flag * 1.0f) + 0.5f);

        std::vector<cv::Point2f> vec;
        vec.emplace_back(p01);
        vec.emplace_back(p10);
        vec.emplace_back(p11);
		
        // 计算最近的坐标点距离采样点的距离
        cv::Point2f distance = p00 - point;
        float len = sqrt(distance.x * distance.x + distance.y * distance.y);

        for (auto p : vec) {
            // point所在的像素已经着色，只需要对剩余的三个像素按照距离远近着色即可
            cv::Point2f d = p - point;
            float l = sqrt(d.x * d.x + d.y * d.y);
            float percent = len / l;

            cv::Vec3d  color = window.at<cv::Vec3b>(p.y, p.x);
            // 取最大值
            color[1] = std::max(color[1], (double)255 * percent);
            window.at<cv::Vec3b>(p.y, p.x) = color;
        }
    }
}

int main() 
{
    cv::Mat window = cv::Mat(700, 700, CV_8UC3, cv::Scalar(0));
    cv::cvtColor(window, window, cv::COLOR_BGR2RGB);
    cv::namedWindow("Bezier Curve", cv::WINDOW_AUTOSIZE);

    cv::setMouseCallback("Bezier Curve", mouse_handler, nullptr);

    int key = -1;
    while (key != 27) 
    {
        for (auto &point : control_points) 
        {
            cv::circle(window, point, 3, {255, 255, 255}, 3);
        }

        if (control_points.size() == 6) 
        {
            // naive_bezier(control_points, window);
            bezier(control_points, window);

            cv::imshow("Bezier Curve", window);
            cv::imwrite("my_bezier_curve.png", window);
            key = cv::waitKey(0);

            return 0;
        }

        cv::imshow("Bezier Curve", window);
        key = cv::waitKey(20);
    }

return 0;
}
