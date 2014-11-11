#pragma once

#include <cmath>
#include <vector>
#include <string>

namespace ol {

	enum Label {
		VEG,
		WIRE,
		POLE,
		GROUND,
		FACADE
	};

    const std::size_t NUM_CLASSES = 5;
    const std::size_t NUM_FEATURES = 10;
    const std::vector<std::string> CLASS_NAMES = {"VEG", "WIRE", "POLE", "GROUND", "FACADE"};

    inline double sigmoid(double val) { return 1.0/(1.0 + std::exp(-val)); }
}
