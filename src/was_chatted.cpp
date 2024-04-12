#include <iostream>
#include <string>
#include <fstream>

#include "argparse.hpp"

using namespace std;

struct MyArgs : public argparse::Args {
    string &model = kwarg("m,model", "The models folder");
    string &data = kwarg("d,data", "The data file to evaluate (1 sample per line)");
    double &alpha = kwarg("a", "The smoothing factor (alpha) value").set_default(1);
};

MyArgs args;

int main(int argc, char* argv[]) {

    args.parse(argc, argv, false);

    // Sample: bits_human | bits_ai -> class
    // 1: 123 | 456 -> human

    return 0;
}