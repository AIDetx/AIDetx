#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <cmath>

#include "argparse.hpp"
#include "json.hpp"

using namespace std;
using json = nlohmann::json;

struct MyArgs : public argparse::Args {
    string &model = kwarg("m,model", "The models folder");
    string &data = kwarg("d,data", "The data file to evaluate (1 sample per line)");
    double &alpha = kwarg("a,alpha", "The smoothing factor (alpha) value").set_default(1);
    bool &verbose = flag("v,verbose", "Print verbose output").set_default(false);
};

MyArgs args;
int k;
int n_chars = 0;
int mapping[256];
double sample_count = 0;
int total_human = 0;
int total_ai = 0;
double fallback_bits;
unordered_map<string, int*> human_model;
unordered_map<string, int*> ai_model;


void sanitize(ifstream &data) {
    if (!data.is_open()) {
        printf("Error: Could not open data file\n");
        exit(1);
    }
    if (args.alpha <= 0 || args.alpha > 1) {
        printf("Error: alpha must be between ]0, 1]\n");
        exit(1);
    }
    if (!filesystem::exists(args.model)) {
        printf("Error: Model folder does not exist\n");
        exit(1);
    }
    if (!filesystem::exists(args.model + "/config.json")) {
        printf("Error: Model folder does not contain config.json\n");
        exit(1);
    }
    if (!filesystem::exists(args.model + "/human.txt")) {
        printf("Error: Model folder does not contain human.txt\n");
        exit(1);
    }
    if (!filesystem::exists(args.model + "/ai.txt")) {
        printf("Error: Model folder does not contain ai.txt\n");
        exit(1);
    }
}


void load_config() {
    ifstream config_file(args.model + "/config.json");
    json config;
    config_file >> config;
    k = config["k"];
    n_chars = config["n_chars"];
    for (int i = 0; i < 256; i++) {
        mapping[i] = -1;
    }
    string alphabet = config["alphabet"];
    for (int i = 0; i < n_chars; i++) {
        mapping[static_cast<unsigned char>(alphabet[i])] = i;
    }
    config_file.close();
    cout << config.dump(4) << endl;
}


void load_model(unordered_map<string, int*> &model, string model_name) {
    ifstream model_file(args.model + "/" + model_name + ".txt");

    string key;
    int value;
    int *counts;
    string line;

    while (getline(model_file, line)) {
        counts = new int[n_chars + 1];
        key = line.substr(0, k);
        line = line.substr(k + 1);
        istringstream iss(line);
        for (int i = 0; i < n_chars + 1; i++) {
            iss >> value;
            counts[i] = value;
        }
        model[key] = counts;
    }
    model_file.close();
    printf("Model %s loaded\n", model_name.c_str());
}


void calculate_fallback() {
    double prob = (0 + args.alpha) / (0 + args.alpha * n_chars);
    fallback_bits = -log2(prob);
}


double calculate_bits(int counts[], char c){
    double prob = (counts[mapping[static_cast<unsigned char>(c)]] + args.alpha) / (counts[n_chars] + args.alpha * n_chars);
    return -log2(prob);
}


double get_bits(unordered_map<string, int*> &model, string key, char c) {
    if (model.find(key) == model.end()) {
        return fallback_bits;
    }
    return calculate_bits(model[key], c);
}


void evaluate_sample(string sample) {

    sample_count++;
    double human_bits = 0;
    double ai_bits = 0;
    string class_;

    if ((int) sample.size() <= k) {
        printf("Error: Sample %.0f is too short\n", sample_count);
        return;
    }

    string clean_sample = "";
    for (char c : sample) {
        if (mapping[static_cast<unsigned char>(c)] != -1) {
            clean_sample += c;
        }
    }
    
    string key = clean_sample.substr(0, k);
    int sample_length = clean_sample.size();
    for (int i = k; i < sample_length; i++) {
        human_bits += get_bits(human_model, key, clean_sample[i]);
        ai_bits += get_bits(ai_model, key, clean_sample[i]);
        key.erase(0, 1);
        key += clean_sample[i];
    }

    if (human_bits > ai_bits) {
        class_ = "AI";
        total_ai++;
    } else {
        class_ = "Human";
        total_human++;
    }

    if (args.verbose) {
        printf("%.0f: %.2f | %.2f -> %s\n", sample_count, human_bits, ai_bits, class_.c_str());
    }
}


int main(int argc, char* argv[]) {

    args.parse(argc, argv, false);
    ifstream data(args.data);

    sanitize(data);
    load_config();
    load_model(human_model, "human");
    load_model(ai_model, "ai");
    calculate_fallback();

    string sample;
    printf("Evaluating samples from %s\n", args.data.c_str());
    while (getline(data, sample)) {
        evaluate_sample(sample);
    }

    printf("\nTotal samples: %.0f\n", sample_count);
    printf("Samples classified as human: %d, %.2f%%\n", total_human, (total_human / sample_count) * 100);
    printf("Samples classified as AI: %d, %.2f%%\n", total_ai, (total_ai / sample_count) * 100);

    return 0;
}