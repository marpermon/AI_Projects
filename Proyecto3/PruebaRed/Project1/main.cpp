#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include "NeuralNetwork.h"
#include <stdexcept>

template <typename T>
std::vector<T> getFirstN(const std::vector<T>& vec, size_t n) {
    if (n > vec.size()) {
        throw std::out_of_range("Requested size exceeds the vector size");
    }
    return std::vector<T>(vec.begin(), vec.begin() + n);
}

template <typename T>
std::vector<T> getLastN(const std::vector<T>& vec, size_t n) {
    if (n > vec.size()) {
        throw std::out_of_range("Requested size exceeds the vector size");
    }
    return std::vector<T>(vec.end() - n, vec.end());
}


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


int main()
{
	std::vector<uint32_t> topology = {6,7,6};
	SimpleNeuralNetwork nn(topology, 0.1);

	std::string inputFile = "features.csv";
    std::vector<std::vector<float>> targetInputs = readCSVToVector(inputFile);

	std::string outputFile = "labels.csv";
    std::vector<std::vector<float>> targetOutputs = readCSVToVector(inputFile);
	size_t rows = vec2D.size();

	
	std::vector<int> first4 = getFirstN(original, 4);

        // Get the last 3 values
    std::vector<int> last3 = getLastN(original, 3);
	uint32_t  epoch = 1000;

	std::cout << "training started\n";

	for (uint32_t i = 0; i < epoch; i++)
	{
		uint32_t index = rand() % 4;
		nn.FeedFordward(targetInputs[index]);
		nn.backPropagate(targetOutputs[index]);
	}

	std::cout << "training completed\n";

	for (auto input : targetInputs)
	{
		nn.FeedFordward(input);
		auto preds = nn.getPrediction();
		std::cout << input[0] << "," << input[1] << " -> " << preds[0]
			<< std::endl;
	}
	return 0;
}