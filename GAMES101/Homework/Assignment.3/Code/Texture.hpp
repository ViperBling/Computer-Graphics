//
// Created by LEI XU on 4/27/19.
//

#ifndef RASTERIZER_TEXTURE_H
#define RASTERIZER_TEXTURE_H
#include "global.hpp"
#include <eigen3/Eigen/Eigen>
#include <opencv2/opencv.hpp>
class Texture{
private:
    cv::Mat image_data;

public:
    Texture(const std::string& name)
    {
        image_data = cv::imread(name);
        cv::cvtColor(image_data, image_data, cv::COLOR_RGB2BGR);
        width = image_data.cols;
        height = image_data.rows;
    }

    int width, height;

    Eigen::Vector3f getColor(float u, float v)
    {
        u = u < 0 ? 0 : u;
        u = u > 1 ? 1 : u;
        v = v < 0 ? 0 : v;
        v = v > 1 ? 1 : v;
        auto u_img = u * width;
        auto v_img = (1 - v) * height;
        auto color = image_data.at<cv::Vec3b>(v_img, u_img);
        return Eigen::Vector3f(color[0], color[1], color[2]);
    }

    Eigen::Vector3f getColorBilinear(float u, float v)
    {
        u = u < 0 ? 0 : u;
        u = u > 1 ? 1 : u;
        v = v < 0 ? 0 : v;
        v = v > 1 ? 1 : v;
        auto u_img = u * width;
        auto v_img = (1 - v) * height;

        // 相当于取一个在u，v处的boundingbox，边界是当前位置的临近4个像素
        auto u_min = std::floor(u_img);
        auto u_max = std::min((float)width, std::ceil(u_img));
        auto v_min = std::floor(v_img);
        auto v_max = std::min((float)height, std::ceil(v_img));

        // 四个临近像素的颜色值，然后进行插值操作
        // opencv中的v方向和贴图中的v方向是反的，v_min实际上在v_max上面
        auto u_00 = image_data.at<cv::Vec3b>(v_max, u_min);
        auto u_10 = image_data.at<cv::Vec3b>(v_max, u_max);
        auto u_01 = image_data.at<cv::Vec3b>(v_min, u_min);
        auto u_11 = image_data.at<cv::Vec3b>(v_min, u_max);

        // 待插值像素在bbx中的位置
        float s = (u_img - u_min) / (u_max - u_min);
        float t = (v_img - v_max) / (v_min - v_max);

        auto u_0 = (1 - s) * u_00 + s * u_10;
        auto u_1 = (1 - s) * u_01 + s * u_11;

        auto F = (1 - t) * u_0 + t * u_1;

        return Eigen::Vector3f(F[0], F[1], F[2]);
    }

};
#endif //RASTERIZER_TEXTURE_H
