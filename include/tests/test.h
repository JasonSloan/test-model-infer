#include "utils/json.hpp"
#include "utils/utilsDef.h"
#include "spdlog/spdlog.h"
#include "yolo/pubDef.h"

using namespace std;

void test_detect(
	string& nickName, string &modelPath, int batch_size, float conf_thre, float nms_thre,
	string device, int max_det, float kpt_visible_thre, string input_dir,
	string channel_id, int max_qsize, int n_iters,
	bool modelLog, bool inferLog, bool use_yuv, bool save_image,
	bool use_callback, bool multi_label, string log_file,
    spdlog::level::level_enum& log_level, LogMode& log_mode,
	bool save_tojson, int saveGap, int saveMax, 
	bool use_custom_timestamp, int timestamp_gap, int sleep_gap
);

void test_track(
	int agg_iters, float conf_thre, float nms_thre,
    int track_label, int fps, int track_buffer,
    float track_thre, float high_thre, float match_thre,
    std::string input_dir, std::string& modelPath,
    int batch_size, int max_det, std::string& device,
    bool modelLog, bool inferLog, bool save_image,
    bool save_video, bool save_tojson,
    int saveGap, int saveMax, std::string save_dir,
	std::string video_save_name, std::string jsonfile_save_name,
    std::string log_file, spdlog::level::level_enum& log_level, LogMode& log_mode
);

void test_countPersons(
	std::string& modelPath, float conf_thre, float nms_thre,
    std::string device, int max_det, int max_qsize,
    std::vector<std::string> sources, std::string savePath,
    bool modelLog, bool inferLog, bool multi_label,
	std::string log_file, spdlog::level::level_enum& log_level,
	LogMode& log_mode, Box& left_restrict, 
	Box& right_restrict, bool draw_restrict, float intrude_thre
);

void test_equalize();

void do_test(nlohmann::json& params_j);