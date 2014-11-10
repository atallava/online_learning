#pragma once
#include <string>

namespace ol{
    class Test{
    public:
        bool testVizPCD(std::string file_name);
        bool testVizPoints();
        bool testDataset(std::string file_name);
        bool validatePredictor(std::string file_name, std::string algo);
    };
}
