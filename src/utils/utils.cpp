#include <dirent.h>  // opendir和readdir包含在这里
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>
#include "json.hpp"
#include "opencv2/opencv.hpp"
#include "spdlog/logger.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include "tqdm.hpp"

#include "utils/utils.h"
#include "utils/utilsDef.h"
#include "yolo/yolo.h"

using namespace std;
using namespace cv;
using json = nlohmann::json;

void setup_logger(const std::string& log_file,
                  spdlog::level::level_enum log_level,
                  LogMode log_mode) {
    std::shared_ptr<spdlog::logger> logger;

    switch (log_mode) {
        case ConsoleSink: {
            auto console_sink =
                std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            logger = std::make_shared<spdlog::logger>("console_logger",
                                                      console_sink);
            break;
        }
        case FileSink: {
            auto file_sink =
                std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file,
                                                                    true);
            logger = std::make_shared<spdlog::logger>("file_logger", file_sink);
            break;
        }
        case MultiSink: {
            auto console_sink =
                std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            auto file_sink =
                std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file,
                                                                    true);
            std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
            logger = std::make_shared<spdlog::logger>(
                "multi_sink", sinks.begin(), sinks.end());
            break;
        }
        default:
            throw std::invalid_argument("Invalid log mode");
    }

    spdlog::set_default_logger(logger);
    spdlog::set_level(log_level);
}

int listdir(string& input, vector<string>& files_vector) {
    DIR* pDir = opendir(input.c_str());
    if (!pDir) {
        cerr << "Error opening directory: " << strerror(errno) << endl;
        return -1;
    }
    struct dirent* ptr;
    while ((ptr = readdir(pDir)) != nullptr) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
            files_vector.push_back(input + "/" + ptr->d_name);
        }
    }
    closedir(pDir);
    std::sort(files_vector.begin(), files_vector.end());
    return 0;
}

bool starts_with(const std::string& str, const std::string& starting) {
    if (str.length() >= starting.length()) {
        return str.compare(0, starting.length(), starting) == 0;
    }
    return false;
}

bool ends_with(const std::string& str, const std::string& ending) {
    if (str.length() >= ending.length()) {
        return str.compare(str.length() - ending.length(), ending.length(),
                           ending) == 0;
    }
    return false;
}

string get_filename(const std::string& file_path, bool with_ext) {
    int index = file_path.find_last_of('/');
    if (index < 0)
        index = file_path.find_last_of('\\');
    std::string tmp = file_path.substr(index + 1);
    if (with_ext)
        return tmp;
    std::string img_name = tmp.substr(0, tmp.find_last_of('.'));
    return img_name;
}

std::string getFileSuffix(const std::string& filePath) {
    size_t dotPos = filePath.find_last_of('.');
    if (dotPos != std::string::npos) {
        return filePath.substr(dotPos + 1);
    }
    return ""; // No suffix found
}

// 画虚线框
void drawDashedRectangle(cv::Mat& image,
                         cv::Point leftTop,
                         cv::Point rightBottom,
                         const cv::Scalar& color,
                         int thickness = 1,
                         int dashLength = 5,
                         int spaceLength = 3) {
    cv::Point rightTop(rightBottom.x, leftTop.y);
    cv::Point leftBottom(leftTop.x, rightBottom.y);
    for (int x = leftTop.x; x < rightTop.x; x += dashLength + spaceLength) {
        cv::line(image, cv::Point(x, leftTop.y),
                 cv::Point(std::min(x + dashLength, rightTop.x), leftTop.y),
                 color, thickness);
    }
    for (int x = leftBottom.x; x < rightBottom.x;
         x += dashLength + spaceLength) {
        cv::line(
            image, cv::Point(x, rightBottom.y),
            cv::Point(std::min(x + dashLength, rightBottom.x), rightBottom.y),
            color, thickness);
    }
    for (int y = leftTop.y; y < leftBottom.y; y += dashLength + spaceLength) {
        cv::line(image, cv::Point(leftTop.x, y),
                 cv::Point(leftTop.x, std::min(y + dashLength, leftBottom.y)),
                 color, thickness);
    }
    for (int y = rightTop.y; y < rightBottom.y; y += dashLength + spaceLength) {
        cv::line(image, cv::Point(rightTop.x, y),
                 cv::Point(rightTop.x, std::min(y + dashLength, rightBottom.y)),
                 color, thickness);
    }
}

void draw_rectangles(vector<Result>& results,
                     vector<Mat>& im0s,
                     vector<string>& save_paths,
                     std::map<int, string>& src_map,
                     bool save) {
    for (int i = 0; i < results.size(); ++i) {
        Result result = results[i];
        Mat& im0 = im0s[i];
        for (int j = 0; j < result.bboxes.size(); j++) {
            auto name = src_map[result.bboxes[j].label];
            float confidence = result.bboxes[j].score;
            cv::Scalar color = COLORS[name];
            int left = result.bboxes[j].left;
            int top = result.bboxes[j].top;
            int right = result.bboxes[j].right;
            int bottom = result.bboxes[j].bottom;
            cv::rectangle(im0, cv::Point(left, top), cv::Point(right, bottom),
                          color, 5, 8, 0);
            auto caption = cv::format("%s %.2f", name.c_str(), confidence);
            int text_width =
                cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(im0, cv::Point(left - 3, top - 33),
                          cv::Point(left + text_width, top), color, -1);
            cv::putText(im0, caption, cv::Point(left, top - 5), 0, 1,
                        cv::Scalar::all(0), 2, 16);
        }
        if (save)
            cv::imwrite(save_paths[i], im0);
    }
}

void draw_rectangles_with_keypoints(vector<Result>& results,
                     vector<Mat>& im0s,
                     vector<string>& save_paths,
                     std::map<int, string>& src_map,
                     float kpt_visible_thre,
                     bool save) {
    for (int i = 0; i < results.size(); ++i) {
        Result result = results[i];
        Mat& im0 = im0s[i];
        for (int j = 0; j < result.bboxes.size(); j++) {
            auto name = src_map[result.bboxes[j].label];
            float confidence = result.bboxes[j].score;
            cv::Scalar color = COLORS[name];
            int left = result.bboxes[j].left;
            int top = result.bboxes[j].top;
            int right = result.bboxes[j].right;
            int bottom = result.bboxes[j].bottom;
            cv::rectangle(im0, cv::Point(left, top), cv::Point(right, bottom),
                          color, 5, 8, 0);
            auto caption = cv::format("%s %.2f", name.c_str(), confidence);
            int text_width =
                cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(im0, cv::Point(left - 3, top - 33),
                          cv::Point(left + text_width, top), color, -1);
            cv::putText(im0, caption, cv::Point(left, top - 5), 0, 1,
                        cv::Scalar::all(0), 2, 16);
            
            // draw keypoints
            auto keypoints = results[i].bboxes[j].keypoints;
            for (int k = 0; k < keypoints.size(); ++k){
                auto kpt_one = keypoints[k];
                bool visible = kpt_one.score > kpt_visible_thre;
                if (visible)
                    cv::circle(im0, cv::Point2d(kpt_one.x, kpt_one.y), 3, cv::Scalar(0, 255, 255), -1);
            }

            // draw skeletons
            if (keypoints.size() != 0)
                for (int s = 0; s < SKELETONS.size(); ++s) {
                    auto pt1 = keypoints[SKELETONS[s][0]];
                    auto pt1_cords = cv::Point2d(pt1.x, pt1.y);
                    auto pt1_visible = pt1.score > kpt_visible_thre;
                    auto pt2 = keypoints[SKELETONS[s][1]];
                    auto pt2_cords = cv::Point2d(pt2.x, pt2.y);
                    auto pt2_visible = pt2.score > kpt_visible_thre;
                    if (pt1_visible && pt2_visible)
                        cv::line(im0, pt1_cords, pt2_cords, cv::Scalar(255, 0, 0), 2);
                }
        }
        if (save)
            cv::imwrite(save_paths[i], im0);
    }
}

void draw_rectangles(Result& result, Mat& im0) {
    for (int j = 0; j < result.bboxes.size(); j++) {
        string name = "person";
        float confidence = result.bboxes[j].score;
        int track_id = result.bboxes[j].track_id;
        cv::Scalar color = COLORS[name];
        int left = result.bboxes[j].left;
        int top = result.bboxes[j].top;
        int right = result.bboxes[j].right;
        int bottom = result.bboxes[j].bottom;
        cv::rectangle(im0, cv::Point(left, top), cv::Point(right, bottom),
                      color, 5, 8, 0);
        auto caption =
            cv::format("%s %.2f %d", name.c_str(), confidence, track_id);
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(im0, cv::Point(left - 3, top - 33),
                      cv::Point(left + text_width, top), color, -1);
        cv::putText(im0, caption, cv::Point(left, top - 5), 0, 1,
                    cv::Scalar::all(0), 2, 16);
    }
}

void draw_rectangles(Result& result, Mat& im0, string& save_path) {
    for (int j = 0; j < result.bboxes.size(); j++) {
        string name = "person";
        float confidence = result.bboxes[j].score;
        int track_id = result.bboxes[j].track_id;
        cv::Scalar color = COLORS[name];
        int left = result.bboxes[j].left;
        int top = result.bboxes[j].top;
        int right = result.bboxes[j].right;
        int bottom = result.bboxes[j].bottom;
        cv::rectangle(im0, cv::Point(left, top), cv::Point(right, bottom),
                      color, 5, 8, 0);
        auto caption =
            cv::format("%s %.2f %d", name.c_str(), confidence, track_id);
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(im0, cv::Point(left - 3, top - 33),
                      cv::Point(left + text_width, top), color, -1);
        cv::putText(im0, caption, cv::Point(left, top - 5), 0, 1,
                    cv::Scalar::all(0), 2, 16);
    }
    cv::imwrite(save_path, im0);
}

void draw_rectangles(std::vector<Result>& results,
                     std::vector<cv::Mat>& im0s,
                     int numPersons,
                     std::map<int, std::string>& src_map,
                     Box& left_restrict,
                     Box& right_restrict,
                     bool draw_restrict) {
    for (int i = 0; i < results.size(); ++i) {
        Result result = results[i];
        Mat& im0 = im0s[i];
        // 画目标检测框
        for (int j = 0; j < result.bboxes.size(); j++) {
            auto name = src_map[result.bboxes[j].label];
            float confidence = result.bboxes[j].score;
            float intrude_ratio = result.bboxes[j].intrude_ratio;
            cv::Scalar color = COLORS[name];
            int left = result.bboxes[j].left;
            int top = result.bboxes[j].top;
            int right = result.bboxes[j].right;
            int bottom = result.bboxes[j].bottom;
            cv::rectangle(im0, cv::Point(left, top), cv::Point(right, bottom),
                          color, 5, 8, 0);
            auto caption1 = cv::format("%s %.2f", name.c_str(), confidence);
            auto caption2 = cv::format("intrude %.2f", intrude_ratio);
            int text_width1 = cv::getTextSize(caption1, 0, 1, 2, nullptr).width;
            int text_width2 = cv::getTextSize(caption2, 0, 1, 2, nullptr).width;
            int text_width = std::max(text_width1, text_width2) + 10;
            int text_height1 =
                cv::getTextSize(caption1, 0, 1, 2, nullptr).height;
            int text_height2 =
                cv::getTextSize(caption2, 0, 1, 2, nullptr).height;
            int total_height = text_height1 + text_height2 + 10;
            cv::rectangle(im0, cv::Point(left - 3, top - total_height - 2),
                          cv::Point(left + text_width, top), color, -1);
            cv::putText(im0, caption1,
                        cv::Point(left, top - total_height + text_height2), 0,
                        1, cv::Scalar::all(0), 2, 16);
            cv::putText(im0, caption2, cv::Point(left, top - 5), 0, 1,
                        cv::Scalar::all(0), 2, 16);
        }

        // 画限制区域框
        if (draw_restrict) {
            drawDashedRectangle(
                im0s[0], cv::Point(left_restrict.left, left_restrict.top),
                cv::Point(left_restrict.right, left_restrict.bottom),
                cv::Scalar(0, 255, 255), 2, 5, 3);
            drawDashedRectangle(
                im0s[1], cv::Point(right_restrict.left, right_restrict.top),
                cv::Point(right_restrict.right, right_restrict.bottom),
                cv::Scalar(0, 255, 255), 2, 5, 3);
        }

        // 把人数写上
        if (i == 1) {  // only put "number persons" in the right camera frame
            int left = 100;
            int top = 300;
            auto caption = cv::format("Number Persons %d", numPersons);
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(
                caption, cv::FONT_HERSHEY_SIMPLEX, 1, 2, &baseline);
            int text_width = textSize.width + 20;
            int text_height = textSize.height + 20;
            cv::rectangle(im0, cv::Point(left, top - text_height),
                          cv::Point(left + text_width, top),
                          cv::Scalar(0, 255, 255), -1);
            cv::putText(im0, caption, cv::Point(left + 10, top - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar::all(0), 2,
                        cv::LINE_AA);
        }
    }
}

void assign_properties(
    std::string input_dir,
    int imgcount,
    int batch_size,
    int n_batch,
    std::vector<std::string>& imgs_paths,
    std::vector<std::vector<std::string>>& imgs_paths_batched,
    std::vector<std::vector<std::string>>& save_paths_batched,
    std::vector<std::vector<std::string>>& unique_ids_batched){
    int total = 0;
    int broken = 0;
    int batch_idx = 0;

    imgs_paths_batched.resize(n_batch);
    save_paths_batched.resize(n_batch);
    unique_ids_batched.resize(n_batch);
    for (int i = 0; i < n_batch * batch_size; ++i) {
        if ((i != 0) && (i % batch_size == 0))
            batch_idx++;
        imgs_paths_batched[batch_idx].push_back(imgs_paths[i]);
        string imgName = get_filename(imgs_paths[i], true);
        string save_path = "outputs/" + imgName;
        save_paths_batched[batch_idx].push_back(save_path);
        string unique_id = get_filename(imgs_paths[i], false);
        unique_ids_batched[batch_idx].push_back(unique_id);
        total++;
    }
    imgcount = total - broken;
    if (imgcount % batch_size != 0) {
        imgcount = imgcount - imgcount % batch_size;
        imgs_paths_batched.pop_back();  
        save_paths_batched.pop_back();
        unique_ids_batched.pop_back();
    }
}

void prepare_oneBatch(
    int i_batch,
    int batch_size,
    std::vector<cv::Mat>& oneBatch_img, 
    Input* oneBatch_input, 
    std::vector<std::vector<std::string>>& imgs_paths_batched,
    std::vector<std::vector<std::string>>& unique_ids_batched,
    long& timestamp,
    bool use_custom_timestamp,
    int timestamp_gap,
    bool use_yuv
){
    for (int i = 0; i < batch_size; ++i){
        cv::Mat img = cv::imread(imgs_paths_batched[i_batch][i], IMREAD_COLOR);
        cv::Mat img_backup = cv::imread(imgs_paths_batched[i_batch][i], IMREAD_COLOR);
        if (use_yuv) cv::cvtColor(img, img, COLOR_BGR2YUV);
        else cv::cvtColor(img, img, COLOR_BGR2RGB);
        oneBatch_img.push_back(img_backup);
        int height = img.rows;
        int width = img.cols;
        int numel = height * width * 3;
        if (use_yuv) numel = height * width * 3 / 2;
        oneBatch_input[i].unique_id = unique_ids_batched[i_batch][i];
        oneBatch_input[i].height = height;
        oneBatch_input[i].width = width;
        oneBatch_input[i].data = new unsigned char[numel];
        oneBatch_input[i].timestamp = getCurrentTimestamp();
        if (use_custom_timestamp) oneBatch_input[i].timestamp = timestamp;
        memcpy(oneBatch_input[i].data, img.data, numel);
    }
    timestamp += timestamp_gap;
}

long getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto epoch = now_ms.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(epoch).count();
}

float mean(vector<float> x) {
    float sum = 0;
    for (int i = 0; i < x.size(); ++i) {
        sum += x[i];
    }
    return sum / x.size();
}

vector<unsigned long> mean(vector<vector<unsigned long long>> ts) {
    vector<unsigned long> interval(ts[0].size() - 1, 0);
    for (int i = 0; i < ts.size(); ++i) {
        for (int j = 0; j < (ts[i].size() - 1); ++j) {
            interval[j] += (ts[i][j + 1] - ts[i][j]);
        }
    }
    for (int k = 0; k < (ts[0].size() - 1); ++k) {
        interval[k] = interval[k] / ts.size();
    }
    return interval;
}

std::string SplitString(const std::string& str, const std::string& delim) {
    std::string out_str;
    std::string::size_type pos1, pos2;
    pos2 = str.find(delim);
    pos1 = 0;
    while (std::string::npos != pos2) {
        pos1 = pos2 + delim.size();
        pos2 = str.find(delim, pos1);
    }
    if (pos1 != str.length())
        out_str = str.substr(pos1);

    return out_str;
}

void removeSubstring(std::string& str, const std::string& substr) {
    size_t pos = str.find(substr);
    if (pos != std::string::npos) {
        str.erase(pos, substr.length());
    }
}

void storeResults(vector<Result>& results,
                  int& saveCount,
                  shared_ptr<vector<Result>>& saveResults) {
    for (int i = 0; i < results.size(); ++i) {
        Result& result_one = results[i];
        saveResults->push_back(result_one);
    }
}

json resultToJson(const Result& result) {
    json j;
    j["channel_id"] = result.channel_id;
    j["unique_id"] = result.unique_id;
    j["timestamp"] = result.timestamp;
    j["msg"] = result.msg;
    j["event_id"] = result.event_id;
    j["event_type"] = result.event_type;
    j["height"] = result.height;
    j["width"] = result.width;
    j["naviStatus"] = result.naviStatus;
    j["data_ref_count"] = result.data_ref_count;
    j["proof_timestamp"] = result.proof_timestamp;

    // Serialize bounding boxes
    for (const auto& bbox : result.bboxes) {
        j["bboxes"].push_back({{"left", bbox.left},
                               {"top", bbox.top},
                               {"right", bbox.right},
                               {"bottom", bbox.bottom},
                               {"score", bbox.score},
                               {"label", bbox.label},
                               {"track_id", bbox.track_id}});
    }

    return j;
}

void saveResultsToJsonFile(
    const std::shared_ptr<std::vector<Result>>& saveResults,
    const std::string& filename) {
    json jArray = json::array();  // Create a JSON array

    // Convert each Result in the vector to JSON and add to the array
    for (const auto& result : *saveResults) {
        jArray.push_back(resultToJson(result));
    }

    // Open the file and write the JSON data
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error: Could not open file " << filename
                  << " for writing." << std::endl;
        return;
    }

    outFile << jArray.dump(
        4);  // Dump with 4 spaces indentation for readability
    outFile.close();
}

int countPersonsWithRestricts(vector<Result>& two_cameras_results,
                              Box& left_restrict,
                              Box& right_restrict,
                              float intrude_thre) {
    assert(two_cameras_results.size() == 2),
        "two camera results supported only";
    vector<Box> restricts = {left_restrict, right_restrict};

    auto intrude_computer = [](Box& bbox, Box& restrict) {
        float cross_left = std::max(bbox.left, restrict.left);
        float cross_top = std::max(bbox.top, restrict.top);
        float cross_right = std::min(bbox.right, restrict.right);
        float cross_bottom = std::min(bbox.bottom, restrict.bottom);
        float cross_area = std::max(0.0f, cross_right - cross_left) *
                           std::max(0.0f, cross_bottom - cross_top);
        float bbox_area = (bbox.right - bbox.left) * (bbox.bottom - bbox.top);

        return cross_area / (bbox_area + 1e-5);
    };

    int totalCounts = 0;
    int maxmuimCount = 0;
    for (int i = 0; i < two_cameras_results.size(); ++i) {
        int n_bboxes = two_cameras_results[i].bboxes.size();
        maxmuimCount = std::max(n_bboxes, maxmuimCount);
        for (int j = 0; j < two_cameras_results[i].bboxes.size(); ++j) {
            // you should filter the bboxes by labels and reserve persons only,
            // here is just a simple example
            float intrude_ratio = intrude_computer(
                two_cameras_results[i].bboxes[j], restricts[i]);
            two_cameras_results[i].bboxes[j].intrude_ratio = intrude_ratio;
            if (intrude_ratio < intrude_thre)
                totalCounts += 1;
        }
    }
    totalCounts = std::max(totalCounts, maxmuimCount);

    return totalCounts;
}