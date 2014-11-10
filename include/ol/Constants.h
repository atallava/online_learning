#pragma once

#include <cmath>

namespace ol {
	enum Label {
		VEG,
		WIRE,
		POLE,
		GROUND,
		FACADE
	};

    const int NUM_FEATURES = 10;

    inline double sigmoid(double val) { return 1.0/(1.0 + std::exp(-val)); }
}