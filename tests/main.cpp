#include <string>
#include <fstream>
#include "utils/json.hpp"
#include "tests/test.h"

using namespace std;
using json = nlohmann::json;

int main() {
	string config_path = "../config.json";
	ifstream config_file(config_path);
	nlohmann::json params_j;
	config_file >> params_j;
	do_test(params_j);
	
    return 0;
};