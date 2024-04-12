#include <iostream>
#include <string>
#include <fstream>

#include "argparse.hpp"

using namespace std;

struct MyArgs : public argparse::Args {
    string &human_data = kwarg("h,human", "The human data file");
    string &ai_data = kwarg("g,gpt", "The AI generated data file");
    string &output = kwarg("o,output", "The output folder to save the models");
    string &alphabet = kwarg("a,alphabet", "The alphabet file if empty, will use the alphabet from the data (converts to lower case)").set_default("");
    int &k = kwarg("k", "k-order Markov model").set_default(5);
    double &alpha = kwarg("a,alpha", "The alpha (smoothing factor) value").set_default(1);
};

MyArgs args;

int main(int argc, char* argv[]) {

    args.parse(argc, argv, false);
    return 0;
}