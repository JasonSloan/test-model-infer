#pragma once

#include <stdio.h>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"  // spdlog日志相关
#include "utils/json.hpp"

#include "utils/utilsDef.h"
#include "yolo/yolo.h"

using json = nlohmann::json;

void setup_logger(const std::string& log_file,
                  spdlog::level::level_enum log_level,
                  LogMode log_mode);

int listdir(std::string& input, std::vector<std::string>& files_vector);

bool starts_with(const std::string& str, const std::string& starting);

bool ends_with(const std::string& str, const std::string& ending);

std::string get_filename(const std::string& file_path, bool with_ext = true);

std::string getFileSuffix(const std::string& filePath);

void draw_rectangles(std::vector<Result>& results,
                     std::vector<cv::Mat>& im0s,
                     std::vector<std::string>& save_paths,
                     std::map<int, std::string>& src_map,
                     bool save = true);

void draw_rectangles(Result& result, cv::Mat& im0);
void draw_rectangles(Result& result, cv::Mat& im0, std::string& save_path);

void draw_rectangles(std::vector<Result>& results,
                     std::vector<cv::Mat>& im0s,
                     int numPersons,
                     std::map<int, std::string>& src_map,
                     Box& left_restrict,
                     Box& right_restrict,
                     bool draw_restrict);

void draw_rectangles_with_keypoints(std::vector<Result>& results,
                     std::vector<cv::Mat>& im0s,
                     std::vector<std::string>& save_paths,
                     std::map<int, std::string>& src_map,
                     float kpt_visible_thre,
                      bool save = true);
                     
void assign_properties(
    std::string input_dir,
    int imgcount,
    int batch_size,
    int n_batch,
    std::vector<std::string>& imgs_paths,
    std::vector<std::vector<std::string>>& imgs_paths_batched,
    std::vector<std::vector<std::string>>& save_paths_batched,
    std::vector<std::vector<std::string>>& unique_ids_batched);

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
    bool use_yuv);

long getCurrentTimestamp();

float mean(std::vector<float> x);

std::vector<unsigned long> mean(
    std::vector<std::vector<unsigned long long>> ts);

std::string SplitString(const std::string& str, const std::string& delim);

void removeSubstring(std::string& str, const std::string& substr);

void storeResults(std::vector<Result>& results,
                  int& saveCount,
                  std::shared_ptr<std::vector<Result>>& saveResults);

json resultToJson(const Result& result);

void saveResultsToJsonFile(
    const std::shared_ptr<std::vector<Result>>& saveResults,
    const std::string& filename);

int countPersonsWithRestricts(std::vector<Result>& two_cameras_results,
                              Box& left_restrict,
                              Box& right_restrict,
                              float intrude_thre);