#include <stdio.h>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <future>
#include "spdlog/logger.h"                              
#include "spdlog/spdlog.h"                             
#include "spdlog/sinks/basic_file_sink.h"              
#include "opencv2/opencv.hpp"
#include "utils/tqdm.hpp"

#include "utils/utilsDef.h"
#include "utils/utils.h"
#include "utils/json.hpp"
#include "yolo/yolo.h"
#include "bytetrack/BYTETracker.h"
#include "yolo/model-utils.h"
#include "tests/test.h"

using namespace std;
using namespace cv;
using json = nlohmann::json;

using time_point = chrono::high_resolution_clock;
template <typename Rep, typename Period>
float micros_cast(const std::chrono::duration<Rep, Period>& d) {
    return static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(d).count()) / 1000.;
}

static bool save_tojson = false;
static int saveCount = 0;
static int saveGap = 5;		// 每saveGap张图存一次result
static int saveMax = 2000;	// 总共存saveMax个result
static bool saved = false;
std::shared_ptr<std::vector<Result>> saveResults = std::make_shared<std::vector<Result>>();
static string savePath = "outputs/results.json";

void callback(std::vector<Result> results, void* userP){
	for (auto& result: results){
		printf("\n");
		spdlog::info("channel_id is {}, image_id is {}, timestamp is {}, bboxes size is {}", 
				result.channel_id, result.unique_id, result.timestamp, result.bboxes.size());
        // spdlog::default_logger()->flush();
        unsigned char* data = result.data;
        delete [] data;

		// int height = result.height;
		// int width = result.width;
		// int numel = height * width * 3;
		// unsigned char* data = result.data;
		// cv::Mat image(height, width, CV_8UC3);
		// memcpy(image.data, data, numel);
		// cvtColor(image, image, cv::COLOR_RGB2BGR);
		// cv::imwrite("recovered.jpg", image);
		// delete [] data;
	}
	if (::save_tojson){
		if (::saveCount < (::saveGap * ::saveMax)){
			if (::saveCount % ::saveGap == 0)
				storeResults(results, ::saveCount, ::saveResults);
		}else{
			if (!::saved){
				saveResultsToJsonFile(::saveResults, ::savePath);
				::saved = true;
			}
		}
		::saveCount++;
	}
}		

// 使用本地图片测试目标检测模型
void test_detect_way1(
	string &nickName,
	string &modelPath, 
	int batch_size,
	float conf_thre,
	float nms_thre,
	string device,
	int max_det, 
	float kpt_visible_thre,
	string input_dir,
	string channel_id,
	int max_qsize,
	int n_iters,
	bool modelLog,
	bool inferLog,
	bool use_yuv, 
	bool save_image,
	bool use_callback,
	bool multi_label,
	string log_file,
	spdlog::level::level_enum& log_level,
	LogMode& log_mode,
	bool use_custom_timestamp, 
	int timestamp_gap,
	int sleep_gap
){
    // setup logger
    setup_logger(log_file, log_level, log_mode);

	//prepare data
	long timestamp = 0;
	vector<string> imgs_paths;
	vector<vector<string>> imgs_paths_batched;
	vector<vector<string>> save_paths_batched;
	vector<vector<string>> unique_ids_batched;
	cv::glob(input_dir + "/*.jpg", imgs_paths);
	int imgcount = imgs_paths.size();
	int n_batch = imgcount / batch_size;
	assign_properties(input_dir, imgcount, batch_size, n_batch, imgs_paths, imgs_paths_batched, save_paths_batched, unique_ids_batched);
	string model_suffix = getFileSuffix(modelPath); string model_classes_suffix = "txt";
	string model_classes_path = replaceSuffix(modelPath, model_suffix, model_classes_suffix);
	std::map<int, string> CURRENT_IDX2CLS = readFileToMap(model_classes_path);
	::saveMax = std::min(::saveMax, imgcount-1);

	// create infer
	shared_ptr<InferInterface> inferController;
	if (!use_callback)
		inferController = create_infer(modelPath, max_det, device, modelLog, multi_label);
	else
		inferController = create_infer(nickName, callback, nullptr, modelPath, max_det, max_qsize, device, modelLog, multi_label);
	inferController->warmup();

	// infer
	spdlog::info("----> Start infering......");
	if (n_iters == -1){
		while (true){
			for (int i : tq::trange(n_batch)){	
				vector<Mat> oneBatch_img; Input oneBatch_input[batch_size];																			// i is batch idx, imgs[i] is number i batch 
				prepare_oneBatch(i, batch_size, oneBatch_img, oneBatch_input, imgs_paths_batched, unique_ids_batched, timestamp, use_custom_timestamp, timestamp_gap, use_yuv);
				if (!use_callback){
					shared_future<vector<Result>> fut = inferController->forward(oneBatch_input, batch_size, conf_thre, nms_thre, inferLog);		// true for log or not
					vector<Result> results = fut.get();																			// asynchronous get result
					if (save_image) draw_rectangles_with_keypoints(results, oneBatch_img, save_paths_batched[i], CURRENT_IDX2CLS, kpt_visible_thre);										// draw rectangles and save image
					delete [] oneBatch_input->data;
				}else{
					inferController->add_images(oneBatch_input, batch_size, conf_thre, nms_thre, channel_id);
					int q_size = inferController->get_qsize();
					spdlog::debug("queue size is {}", q_size);
					this_thread::sleep_for(std::chrono::milliseconds(sleep_gap));
				}
			}
		}
	}else{
		for (int n = 0; n < n_iters; ++n) {
			for (int i : tq::trange(n_batch)){	
				vector<Mat> oneBatch_img; Input oneBatch_input[batch_size];																			// i is batch idx, imgs[i] is number i batch 
				prepare_oneBatch(i, batch_size, oneBatch_img, oneBatch_input, imgs_paths_batched, unique_ids_batched, timestamp, use_custom_timestamp, timestamp_gap, use_yuv);
				if (!use_callback){
					shared_future<vector<Result>> fut = inferController->forward(oneBatch_input, batch_size, conf_thre, nms_thre, inferLog);		// true for log or not
					vector<Result> results = fut.get();																			// asynchronous get result
					if (save_image) draw_rectangles_with_keypoints(results, oneBatch_img, save_paths_batched[i], CURRENT_IDX2CLS, kpt_visible_thre);										// draw rectangles and save image
					delete [] oneBatch_input->data;
				}else{
					inferController->add_images(oneBatch_input, batch_size, conf_thre, nms_thre, channel_id);
					int q_size = inferController->get_qsize();
					spdlog::debug("queue size is {}", q_size);
					this_thread::sleep_for(std::chrono::milliseconds(sleep_gap));
				}
			}
		}
	}
	spdlog::info("----> Infer successfully!");

	// print elapsed time
	auto records = inferController->get_records();
	auto avg_preprocess_time = mean(records[0]) / float(batch_size);
	auto avg_infer_time = mean(records[1]) / float(batch_size);
	auto avg_postprocess_time = mean(records[2]) / float(batch_size);
	auto avg_total_time = avg_preprocess_time + avg_infer_time + avg_postprocess_time;
	spdlog::info("----> Average time cost: preprocess {} ms, infer {} ms, postprocess {} ms, total {} ms", 
		avg_preprocess_time, avg_infer_time, avg_postprocess_time, avg_total_time);

	printf("\n");
	spdlog::warn("Main thread sleep......");
	std::this_thread::sleep_for(std::chrono::seconds(10000));
}

// 使用rtsp视频流或者视频测试目标检测模型
void test_detect_way2(
	string &nickName,
	string &modelPath, 
	int batch_size,
	float conf_thre,
	float nms_thre,
	string device,
	int max_det, 
	float kpt_visible_thre,
	string input_dir,
	string channel_id,
	int max_qsize,
	bool modelLog,
	bool inferLog,
	bool use_yuv, 
	bool save_image,
	bool use_callback,
	bool multi_label,
	string log_file,
	spdlog::level::level_enum& log_level,
	LogMode& log_mode,
	bool use_custom_timestamp, 
	int timestamp_gap,
	int sleep_gap
){
    // setup logger
    setup_logger(log_file, log_level, log_mode);

	//prepare rectified map and videoCapture
	int imgcount = 0; 
	long timestamp = 0;
	auto cap = cv::VideoCapture(input_dir);
	string model_suffix = "xml"; string model_classes_suffix = "txt";
	string model_classes_path = replaceSuffix(modelPath, model_suffix, model_classes_suffix);
	std::map<int, string> CURRENT_IDX2CLS = readFileToMap(model_classes_path);

	// create infer
	shared_ptr<InferInterface> inferController;
	if (!use_callback)
		inferController = create_infer(modelPath, max_det, device, modelLog, multi_label);
	else
		inferController = create_infer(nickName, callback, nullptr, modelPath, max_det, max_qsize, device, modelLog, multi_label);
	inferController->warmup();

	// infer
	spdlog::info("----> Start infering......");
	while(1){
		Input i_imgs[1][batch_size]; 
        vector<vector<Mat>> m_imgs(1);
        vector<vector<string>> save_paths(1);
		for (int i = 0; i < batch_size;){
			bool success; cv::Mat frame;
			success = cap.read(frame);
			if (!success) continue;
            cvtColor(frame, frame, cv::COLOR_BGR2RGB);	// 马哥给的是RGB不是BGR的图
			int height = frame.rows;
			int width = frame.cols;
			int numel = height * width * 3;
			Input img_one;
			img_one.unique_id = std::to_string(imgcount);
			img_one.height = height;
			img_one.width = width;
			img_one.data = new unsigned char[numel];
			img_one.timestamp = getCurrentTimestamp();
			if (use_custom_timestamp) img_one.timestamp = timestamp;
			memcpy(img_one.data, frame.data, numel);
			i_imgs[0][i] = img_one;
            m_imgs[0].push_back(frame);
			string save_path = "outputs/" + std::to_string(imgcount) + ".jpg";
            save_paths[0].push_back(save_path);
            imgcount++; i++;
		}
		timestamp += timestamp_gap;

        // 如果不使用callback接口, 那么调用结束后直接删除图片数据; 如果走callback接口, 那么在callback回调函数中删除图片数据
        bool overflow = false;
		if (!use_callback){
			shared_future<vector<Result>> fut = inferController->forward(i_imgs[0], batch_size, conf_thre, inferLog);		// true for log or not
			vector<Result> results = fut.get();																			// asynchronous get result
			if (save_image) draw_rectangles(results, m_imgs[0], save_paths[0], CURRENT_IDX2CLS);										// draw rectangles and save image
            for (int i = 0; i < batch_size; ++i)
                delete i_imgs[0][i].data;
		}else{
		    overflow = inferController->add_images(i_imgs[0], batch_size, conf_thre, nms_thre, channel_id);
			int q_size = inferController->get_qsize();
			spdlog::debug("queue size is {}", q_size);
			this_thread::sleep_for(std::chrono::milliseconds(sleep_gap));
		}
	}

	// print elapsed time
	auto records = inferController->get_records();
	auto avg_preprocess_time = mean(records[0]) / float(batch_size);
	auto avg_infer_time = mean(records[1]) / float(batch_size);
	auto avg_postprocess_time = mean(records[2]) / float(batch_size);
	auto avg_total_time = avg_preprocess_time + avg_infer_time + avg_postprocess_time;
	spdlog::info("----> Average time cost: preprocess {} ms, infer {} ms, postprocess {} ms, total {} ms", 
		avg_preprocess_time, avg_infer_time, avg_postprocess_time, avg_total_time);

	printf("\n");
	spdlog::warn("Main thread sleep......");
	std::this_thread::sleep_for(std::chrono::seconds(10000));
}

void test_detect(
	string& nickName, string &modelPath, int batch_size, float conf_thre, float nms_thre,
	string device, int max_det, float kpt_visible_thre, string input_dir,
	string channel_id, int max_qsize, int n_iters,
	bool modelLog, bool inferLog, bool use_yuv, bool save_image,
	bool use_callback, bool multi_label, string log_file,
    spdlog::level::level_enum& log_level, LogMode& log_mode,
	bool save_tojson, int saveGap, int saveMax, 
	bool use_custom_timestamp, int timestamp_gap, int sleep_gap
){
	::save_tojson = save_tojson;
	::saveGap = saveGap;
	::saveMax = saveMax;
	bool is_rtsp = starts_with(input_dir, "rtsp");
	bool is_video = ends_with(input_dir, "mp4") || ends_with(input_dir, "mov");
    if (!starts_with(input_dir, "rtsp") && !is_video)
        test_detect_way1(nickName, modelPath, batch_size, conf_thre, nms_thre, device, max_det, kpt_visible_thre, input_dir, channel_id, 
        max_qsize, n_iters, modelLog, inferLog, use_yuv, save_image, use_callback, multi_label, log_file, log_level, log_mode,
		use_custom_timestamp, timestamp_gap, sleep_gap);
    else
        test_detect_way2(nickName, modelPath, batch_size, conf_thre, nms_thre, device, max_det, kpt_visible_thre, input_dir, channel_id,
        max_qsize, modelLog, inferLog, use_yuv, save_image, use_callback, multi_label, log_file, log_level, log_mode,
		use_custom_timestamp, timestamp_gap, sleep_gap);
}

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
){
	// create infer
	auto inferController = create_infer(modelPath, max_det, device, modelLog);
	inferController->warmup();
	// create tracker
	BYTETracker tracker(fps, track_buffer, track_thre, high_thre, match_thre);
	// prepare logger
    setup_logger(log_file, log_level, log_mode);
	// prepare data
	int count = 0;
	string images_save_path;
	string json_save_path;
	bool json_saved = false;
	int jsonSaveCount = 0;
	std::shared_ptr<std::vector<Result>> saveResults = std::make_shared<std::vector<Result>>();
	Input i_imgs[1];
	int total_frames;
	vector<cv::String> frame_names;
	vector<float> track_records;    // 对跟踪计时
	int cap_fps;
	cv::VideoCapture cap;
	cv::VideoWriter videoWriter;
	bool is_video = ends_with(input_dir, ".mp4") || ends_with(input_dir, ".mov") || starts_with(input_dir, "rtsp");
	if (!is_video){
		cv::glob(input_dir, frame_names, false);
		total_frames = frame_names.size();
	}else{
		cap = cv::VideoCapture(input_dir);
		if (!cap.isOpened()) {
			std::cerr << "Cant open file" << input_dir << "!" << std::endl;
			return;
		}
		total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
		if (starts_with(input_dir, "rtsp") || total_frames == -1) total_frames = 100000;
		if (save_video) {
			string video_save_path = save_dir + "/" + video_save_name;
			int codec = cv::VideoWriter::fourcc('X', '2', '6', '4'); // MJPG for avi format video
			cv::Mat frame;
			cap_fps = cap.get(cv::CAP_PROP_FPS);
			bool success = cap.read(frame);
			auto frameWidth = frame.cols;
			auto frameHeight = frame.rows;
			videoWriter = cv::VideoWriter(video_save_path, codec, cap_fps, cv::Size(frameWidth, frameHeight));
			if (!videoWriter.isOpened()) {
				std::cerr << "Cant open file" << video_save_path << "!" << std::endl;
				return;
			}
		}
	}

	// start track
    for (int i : tq::trange(total_frames)){
        count++;
        cv::Mat frame; bool success = true;
        if (!is_video){
            frame = cv::imread(frame_names[i]);
        }else{
            success = cap.read(frame);
        }
        if (!success) continue;

        cvtColor(frame, frame, cv::COLOR_BGR2RGB);	// 马哥给的是RGB不是BGR的图
        int height = frame.rows;
        int width = frame.cols;
        int numel = height * width * 3;
        i_imgs[0].height = height;
        i_imgs[0].width = width;
        i_imgs[0].data = new unsigned char[numel];
        i_imgs[0].unique_id = std::to_string(i);
		if (!is_video) i_imgs[0].unique_id = get_filename(frame_names[i]);
        i_imgs[0].timestamp = getCurrentTimestamp();
        memcpy(i_imgs[0].data, frame.data, numel);

        auto results = inferController->forward(i_imgs, batch_size, conf_thre, nms_thre, inferLog).get();

        vector<Object> objects;
        auto start = time_point::now();
        int n_bboxes = results[0].bboxes.size();
        for (int j = 0; j < n_bboxes; ++j){
			if (results[0].bboxes[j].label != track_label) continue;
            Object object;
            float left = results[0].bboxes[j].left;
            float top = results[0].bboxes[j].top;
            float right = results[0].bboxes[j].right;
            float bottom = results[0].bboxes[j].bottom;
            float w = right - left;
            float h = bottom - top;
            ObRect rect = {left, top, w, h};
            object.rect = rect;
            object.prob = results[0].bboxes[j].score;
            object.label = results[0].bboxes[j].label;
            objects.push_back(object);
        }
        // auto tracks = tracker.update(objects, count, agg_iters);
		auto tracks = tracker.update(objects);
		results[0].bboxes.clear();
		for (int i = 0; i < tracks.size(); ++i){
			Box box_one;
			auto &track = tracks[i];
			box_one.left = (int)track.tlbr[0];
			box_one.top = (int)track.tlbr[1];
			box_one.right = (int)track.tlbr[2];
			box_one.bottom = (int)track.tlbr[3];
			box_one.score = track.score;
			box_one.label = track.label;
			box_one.track_id = track.track_id;
			results[0].bboxes.push_back(box_one);
		}
        auto stop = time_point::now();
        track_records.push_back(micros_cast(stop - start));
        if (track_records.size() == agg_iters){
            spdlog::info("Last {} iters average time cost: {} ms", agg_iters, mean(track_records));
            spdlog::info("-------------------------------------------------------------------------------");
            track_records.clear();
        }
		if (log_mode == LogMode::FileSink || log_mode == LogMode::MultiSink)
        	spdlog::default_logger()->flush();

        // save image or save_video
        if (save_image || save_video){
            if (!is_video) {
                images_save_path = "outputs/" + get_filename(frame_names[i]);
            }else{
                images_save_path = "outputs/" + std::to_string(count) + ".jpg";
            }

			if (save_image){
				draw_rectangles(results[0], frame, images_save_path);
			}else{
				draw_rectangles(results[0], frame);
				videoWriter.write(frame);
			}
        }

		// save track results to json
		if (save_tojson){
			json_save_path = save_dir + jsonfile_save_name;
			if (jsonSaveCount < (saveGap * saveMax)){
				if (jsonSaveCount % saveGap == 0)
					storeResults(results, jsonSaveCount, saveResults);
			}else{
				if (!json_saved){
					saveResultsToJsonFile(saveResults, json_save_path);
					json_saved = true;
				}
			}
			jsonSaveCount++;
		}

        delete [] i_imgs[0].data;
    }
	videoWriter.release();
	if (save_tojson)
		if (!json_saved)
			saveResultsToJsonFile(saveResults, json_save_path);
}

void test_countPersons(
	std::string& modelPath, float conf_thre, float nms_thre,
    std::string device, int max_det, int max_qsize,
    std::vector<std::string> sources, std::string savePath,
    bool modelLog, bool inferLog, bool multi_label,
	std::string log_file, spdlog::level::level_enum& log_level,
	LogMode& log_mode, Box& left_restrict, 
	Box& right_restrict, bool draw_restrict, float intrude_thre
){
    // setup logger
    setup_logger(log_file, log_level, log_mode);

	//prepare videoCapture
	int imgcount = 0; int batch_size = 1;
	assert (sources.size() == 2), "two cameras supported only!";
	vector<cv::VideoCapture> caps;
	int fps; int frameWidth; int frameHeight; double total_frames;
	for (int i = 0; i < 2; ++i){
		auto cap = cv::VideoCapture(sources[i]);
		caps.push_back(cap);
		fps = cap.get(CAP_PROP_FPS);
		total_frames = cap.get(CAP_PROP_FRAME_COUNT);
		cv::Mat frame;
		bool success = caps[i].read(frame);
		frameWidth = frame.cols;
		frameHeight = frame.rows;
	} 
	int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
	cv::VideoWriter videoWriter(savePath, codec, fps, cv::Size(frameWidth * 2, frameHeight));
	if (!videoWriter.isOpened()) {
		std::cerr << "Cant open the local video file!" << std::endl;
		return;
	}

	// some other preparations
	string model_suffix = "xml"; string model_classes_suffix = "txt";
	string model_classes_path = replaceSuffix(modelPath, model_suffix, model_classes_suffix);
	std::map<int, string> CURRENT_IDX2CLS = readFileToMap(model_classes_path);

	// create infer
	shared_ptr<InferInterface> inferController;
	inferController = create_infer(modelPath, max_det, device, modelLog, multi_label);
	inferController->warmup();

	// infer
	spdlog::info("----> Start infering......");
	bool success = true; if (total_frames == -1) total_frames = 10000;
	for (int i : tq::trange(total_frames)){
		vector<cv::Mat> frames(2); cv::Mat concatenated;
		vector<Result> two_cameras_results_cache;
		int numPersons = 0;
		for (int i = 0; i < 2; ++i){
			Input i_imgs[1][1]; 
			vector<vector<string>> save_paths(1);
			success = caps[i].read(frames[i]);
			if (!success) 
				goto end_loops;
			cvtColor(frames[i], frames[i], cv::COLOR_BGR2RGB);	// 马哥给的是RGB不是BGR的图
			int height = frames[i].rows;
			int width = frames[i].cols;
			int numel = height * width * 3;
			Input img_one;
			img_one.unique_id = std::to_string(imgcount++);
			img_one.height = height;
			img_one.width = width;
			img_one.data = new unsigned char[numel];
			img_one.timestamp = getCurrentTimestamp();
			memcpy(img_one.data, frames[i].data, numel);
			i_imgs[0][0] = img_one;

			shared_future<vector<Result>> fut = inferController->forward(i_imgs[0], batch_size, conf_thre, nms_thre, inferLog);		
			vector<Result> results = fut.get();	
			two_cameras_results_cache.push_back(results[0]);	// two_cameras_results_cache:[left_camera_result, right_camera_result]
			if (i == 1) {
				numPersons = countPersonsWithRestricts(
								two_cameras_results_cache, left_restrict,
								right_restrict, intrude_thre
							);
				draw_rectangles(two_cameras_results_cache, frames, numPersons, CURRENT_IDX2CLS, left_restrict, right_restrict, draw_restrict);
				two_cameras_results_cache.clear();
			}																	
			delete i_imgs[0][0].data;
			spdlog::default_logger()->flush();	
		}	
		cv::hconcat(frames[0], frames[1], concatenated);
		videoWriter.write(concatenated);
	}
	end_loops:	
		videoWriter.release();
		
	printf("\n");
	spdlog::warn("Main thread sleep......");
	std::this_thread::sleep_for(std::chrono::seconds(10000));
}

void test_equalize(){
	int n_iters = 1;
	string img_path = "inputs/images/0002_1_000002.jpg";
	cv::Mat img = cv::imread(img_path);
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < n_iters; ++i){
		cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
		cv::equalizeHist(img, img);
		cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto avg_duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count() / 1000. / n_iters;
	spdlog::info("Avg time cost for {} iterations is {}\n", n_iters, avg_duration);
}

void do_test(json& params_j){
	//------------------------初始配置------------------------
	auto test_type = TestTypeMap.at(params_j["test_type"]);		// 测试类型(检测还是跟踪)
	auto log_mode = LogModeMap.at(params_j["log_mode"]);		// 日志模式(只打印在console还是只写入日志文件还是都要)
	auto log_level = LogLevelMap.at(params_j["log_level"]);	// 日志等级(低于该等级的日志不打印)

    //------------检测配置[使用图片、视频或者rtsp流测试目标检测(可单batch或者多batch)]------------
	if (test_type == TestType::Detect){
		json detect_params_j = params_j["Detect"];

		string nickName 				= detect_params_j["nickName"];
		string modelPath_common 		= detect_params_j["modelPath_common"];
		int batch_size 					= detect_params_j["batch_size"];
		float conf_thre 				= detect_params_j["conf_thre"];
		float nms_thre 					= detect_params_j["nms_thre"];
		string device 					= detect_params_j["device"];
		int max_det 					= detect_params_j["max_det"];
		float keypoint_visible_thre 	= detect_params_j["keypoint_visible_thre"];

		string input_dir 				= detect_params_j["input_dir"];
		string channel_id 				= detect_params_j["channel_id"];
		int max_qsize 					= detect_params_j["max_qsize"];
		int n_iters 					= detect_params_j["n_iters"];

		string log_file 				= detect_params_j["log_file"];
		
		bool modelLog 					= detect_params_j["modelLog"];
		bool inferLog 					= detect_params_j["inferLog"];
		bool use_yuv 					= detect_params_j["use_yuv"];	// 是使用yuv数据作为推理代码的输入数据还是RGB为输入数据
		bool save_image 				= detect_params_j["save_image"];	// 是否将框画在图片上后到本地(为true的话, 每一帧都会保存到本地)
		bool save_tojson 				= detect_params_j["save_tojson"];	// 是否保存results结构体到json文件
		int saveGap 					= detect_params_j["saveGap"];			// 每saveGap张图存一次result(save_tojson的二级配置)
		int saveMax 					= detect_params_j["saveMax"];		// 总共存saveMax个result(save_tojson的二级配置)
		bool use_custom_timestamp 		= detect_params_j["use_custom_timestamp"]; // 是直接获取本地时间戳还是自己手动设置时间戳
		int timestamp_gap 				= detect_params_j["timestamp_gap"];		// 如果是自己手动设置时间戳, 那么没两张图之间的时间差
		int sleep_gap 					= detect_params_j["sleep_gap"];     // 每两张图推理之间实际sleep了多长时间, 在192.168.103.120的设备上用openvino每两张图的推理时间大概是40多ms
		bool use_callback 				= detect_params_j["use_callback"]; // true: 使用add_images接口; false: 直接使用forward接口
		bool multi_label 				= detect_params_j["multi_label"];
		vector<string> modelPaths = {modelPath_common};
		
		test_detect(nickName, modelPath_common, batch_size, conf_thre, nms_thre, device, max_det, keypoint_visible_thre,
					input_dir, channel_id, max_qsize, n_iters, modelLog, inferLog, use_yuv,
					save_image, use_callback, multi_label, log_file, log_level, log_mode,
					save_tojson, saveGap, saveMax, use_custom_timestamp, timestamp_gap, sleep_gap);
	}

    //------------跟踪配置[使用图片、视频或者rtsp流测试目标跟踪(只能单batch)]------------
	if (test_type == TestType::Track){
		json track_params_j = params_j["Track"];

		int agg_iters 					= track_params_j["agg_iters"];	// 每agg_iters算一下跟踪的平均耗时
		int track_label 				= track_params_j["track_label"]; 	// 要跟踪的类别索引
		int fps 						= track_params_j["fps"];
		int track_buffer 				= track_params_j["track_buffer"];
		float track_thre 				= track_params_j["track_thre"];
		float high_thre 				= track_params_j["high_thre"];
		float match_thre			 	= track_params_j["match_thre"];

		string modelPath_common 		= track_params_j["modelPath_common"];
		int batch_size 					= track_params_j["batch_size"];
		float conf_thre 				= track_params_j["conf_thre"];
		float nms_thre 					= track_params_j["nms_thre"];
		string device 					= track_params_j["device"];
		int max_det 					= track_params_j["max_det"];
		string input_dir 				= track_params_j["input_dir"];
		bool use_yuv 					= track_params_j["use_yuv"];	// 是使用yuv数据作为推理代码的输入数据还是RGB为输入数据
		string log_file 				= track_params_j["log_file"];
		bool modelLog 					= track_params_j["modelLog"];
		bool inferLog 					= track_params_j["inferLog"];
		bool save_image 				= track_params_j["save_image"];
		bool save_video 				= track_params_j["save_video"];
		bool save_tojson 				= track_params_j["save_tojson"];
		int saveGap 					= track_params_j["saveGap"];			// 每saveGap张图存一次result(save_tojson的二级配置)
		int saveMax 					= track_params_j["saveMax"];		// 总共存saveMax个result(save_tojson的二级配置)
		string save_dir 				= track_params_j["save_dir"];
		string video_save_name 			= track_params_j["video_save_name"];
		string jsonfile_save_name 		= track_params_j["jsonfile_save_name"];	// 把跟踪结果存在json文件里

		test_track(agg_iters, conf_thre, nms_thre, track_label, fps, track_buffer, track_thre,
				high_thre, match_thre, input_dir, modelPath_common, batch_size,
				max_det, device, modelLog, inferLog, save_image, save_video, save_tojson, saveGap, saveMax, 
				save_dir, video_save_name, jsonfile_save_name, log_file, log_level, log_mode);
	}

	//--------业务算法, 两路视频联合起来数人数, 主要针对的是晚上左摄像头只能检测到左面的人, 右摄像头只能检测到右面的人的情况的测试--------
	// 使用视频或者rtsp流测试夜晚2个摄像头联合数人头(只能单batch)
	if (test_type == TestType::CountPersons){
		json countp_params_j = params_j["CountPersons"];

		vector<string> sources 			= {countp_params_j["sources"][0], countp_params_j["sources"][1]};	// 注意这里一定要左摄像头在前, 右摄像头在后
		string savePath 				= countp_params_j["savePath"];

		Box left_restrict, right_restrict;
		left_restrict.left 				= countp_params_j["left_restrict"]["left"]; 	
		left_restrict.top  				= countp_params_j["left_restrict"]["top"]; 
		left_restrict.right  			= countp_params_j["left_restrict"]["right"];  
		left_restrict.bottom  			= countp_params_j["left_restrict"]["bottom"];
		right_restrict.left 			= countp_params_j["right_restrict"]["left"];	
		right_restrict.top 				= countp_params_j["right_restrict"]["top"];	
		right_restrict.right 			= countp_params_j["right_restrict"]["right"]; 
		right_restrict.bottom 			= countp_params_j["right_restrict"]["bottom"];
		bool draw_restrict 				= countp_params_j["draw_restrict"]; 
		float intrude_thre 				= countp_params_j["intrude_thre"];	// 检测框占区域的多少百分比才会忽略该检测框

		string modelPath_common 		= countp_params_j["modelPath_common"];
		float conf_thre 				= countp_params_j["conf_thre"];
		float nms_thre 					= countp_params_j["nms_thre"];
		string device 					= countp_params_j["device"];
		int max_det 					= countp_params_j["max_det"];
		int max_qsize 					= countp_params_j["max_qsize"];

		bool use_yuv 					= countp_params_j["use_yuv"];
		string log_file 				= countp_params_j["log_file"];
		bool modelLog 					= countp_params_j["modelLog"];
		bool inferLog 					= countp_params_j["inferLog"];
		bool multi_label 				= countp_params_j["multi_label"];
		vector<string> modelPaths = {modelPath_common};
		
		test_countPersons(modelPath_common, conf_thre, nms_thre, device, max_det, max_qsize,
						sources, savePath, modelLog, inferLog, multi_label,
						log_file, log_level, log_mode, 
						left_restrict, right_restrict, draw_restrict, intrude_thre);
	}

	//--------------------------------------------------------直方图均衡化--------------------------------------------------------
	if (test_type == TestType::HistEqualize)			
    	test_equalize();
}