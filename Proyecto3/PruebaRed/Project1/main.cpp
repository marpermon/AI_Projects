#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include "NeuralNetwork.h"
#include <stdexcept>
#include <iomanip>



std::vector<std::vector<float>> readCSVToVector(const std::string& filename) {
    std::vector<std::vector<float>> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue; // Skip empty lines

        std::vector<float> row;
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ',')) {
            try {
                row.push_back(std::stof(value)); // Convert each value to float
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid value '" << value << "' in file " << filename << std::endl;
                return {}; // Return an empty vector on failure
            }
        }

        if (!row.empty()) {
            data.push_back(row); // Add valid row to the data
        }
    }

    file.close();
    return data;
}


void genData(std::string filename)
{
    std::ofstream file1(filename + "-in.csv");
    std::ofstream file2(filename + "-out.csv");
    for (uint r = 0; r < 5000; r++) {
        float x = rand() / float(RAND_MAX);
        float y = rand() / float(RAND_MAX);
        file1 << x << ", " << y << std::endl;
        file2 << 2 * x + 10 + y << std::endl;
    }
    file1.close();
    file2.close();
}

int main()
{   
    size_t n;
    genData("test");

	std::vector<uint32_t> topology = {2,3,1};
	SimpleNeuralNetwork nn(topology, 0.1);

	std::string inputFile = "test-in.csv";
    std::vector<std::vector<float>> targetInputs = readCSVToVector(inputFile);
    n = round(targetInputs.size()*0.8);
    std::vector<std::vector<float>> targetInputsTrain(targetInputs.begin(), targetInputs.begin() + n);

    n = round(targetInputs.size()*0.2);
    std::vector<std::vector<float>> targetInputsValidate(targetInputs.begin(), targetInputs.begin() + n);

	std::string outputFile = "test-out.csv";
    std::vector<std::vector<float>> targetOutputs = readCSVToVector(inputFile);
    n = round(targetOutputs.size()*0.8);
    std::vector<std::vector<float>> targetOutputsTrain(targetOutputs.begin(), targetOutputs.begin() + n);
	
    n = round(targetOutputs.size()*0.2);
    std::vector<std::vector<float>> targetOutputsValidate(targetOutputs.begin(), targetOutputs.begin() + n);
	
	uint32_t  epoch = 10000;
    uint32_t  k = 0;

	std::cout << "training started\n";

	for (uint32_t i = 0; i < epoch; i++)
	{
		uint32_t index = rand() % 4;
		nn.FeedFordward(targetInputsTrain[index]);
		nn.backPropagate(targetOutputsTrain[index]);
        std::cout << "\rprogress : " << static_cast<double>(i) * 100.0 / static_cast<double>(epoch) << " %" << std::flush;
	}

	std::cout << "training completed\n";

	for (auto input : targetInputsValidate)
	{
		nn.FeedFordward(input);
		auto preds = nn.getPrediction();
        for(int i=0; i < input.size()-1; i++){
            std::cout << std::left << std::setw(10) << input[i] << ", ";
        }
        std::cout << std::left << std::setw(10) << input[input.size()-1];

        std::cout << " -> " ;
        
        std::vector<float> real = targetOutputsValidate[k];

        float error = 0;
        std::cout << "Predicted = " << std::left << std::setw(10) << preds[0] << ", Real =" << std::setw(10) << real[0];
        std::cout <<  " Diff = " << std::left << std::abs(preds[0] - real[0])*100.0/real[0] << "%\n";        
        k++;      
	}
	return 0;
}