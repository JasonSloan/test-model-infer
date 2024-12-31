#pragma once

#include <string>
#include <vector>
#include <map>
#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

enum TestType {
	Detect,			// 检测
	Track,			// 跟踪
	CountPersons,	// 业务算法, 两路视频联合起来数人数, 主要针对的是晚上左摄像头只能检测到左面的人, 右摄像头只能检测到右面的人的情况的测试
	HistEqualize
};

enum LogMode{
	ConsoleSink,	// 日志只打印在console中
	FileSink,		// 日志只输出到文件中
	MultiSink		// 日志既打印在console中也输出到文件中
};

static std::map<std::string, TestType> TestTypeMap = {
	{"Detect", 			TestType::Detect},
	{"Track",  			TestType::Track},
	{"CountPersons", 	TestType::CountPersons},
	{"HistEqualize", 	TestType::HistEqualize},
};

static std::map<std::string, LogMode> LogModeMap = {
	{"ConsoleSink", LogMode::ConsoleSink},
	{"FileSink",  	LogMode::FileSink},
	{"MultiSink", 	LogMode::MultiSink},
};

static std::map<std::string, spdlog::level::level_enum> LogLevelMap = {
	{"debug", 	spdlog::level::debug},
	{"info",  	spdlog::level::info},
	{"warn", 	spdlog::level::warn},
	{"err", 	spdlog::level::err},
};

static std::map<std::string, cv::Scalar> COLORS = {
	{"person", 				{0, 0, 255}},
	{"person_uniform", 		{0, 255, 255}},
	{"person_ununiform", 	{0, 0, 255}},
	{"head", 				{0, 255, 255}},
	{"helmet", 				{255, 128, 0}},
	{"lookout", 			{51, 255, 51}}
};

static std::vector<std::vector<int>> SKELETONS = {
	{15, 13},
	{13, 11},
	{16, 14},
	{14, 12},
	{11, 12},
	{5, 11},
	{6, 12},
	{5, 6},
	{5, 7},
	{6, 8},
	{7, 9},
	{8, 10},
	{1, 2},
	{0, 1},
	{0, 2},
	{1, 3},
	{2, 4},
	{3, 5},
	{4, 6},
};


// static std::vector<cv::Scalar> COLORS = {
// 		{255, 128, 0},
// 		{255, 153, 51},
// 		{255, 178, 102},
// 		{230, 230, 0},
// 		{255, 153, 255},
// 		{153, 204, 255},
// 		{255, 102, 255},
// 		{255, 51, 255},
// 		{102, 178, 255},
// 		{51, 153, 255},
// 		{255, 153, 153},
// 		{255, 102, 102},
// 		{255, 51, 51},
// 		{153, 255, 153},
// 		{102, 255, 102},
// 		{51, 255, 51},
// 		{0, 255, 0},
// 		{0, 0, 255},
// 		{255, 0, 0},
// 		{255, 255, 255}	
// };