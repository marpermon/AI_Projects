#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include "NeuralNetwork.h"
#include <stdexcept>
#include <iomanip>
#include <random>

// Function to generate a random number in the range [min, max]
uint32_t getRandomNumber(uint32_t min, uint32_t max) {
    // Create a random device and seed
    std::random_device rd;
    std::mt19937 generator(rd()); // Use Mersenne Twister for random number generation
    std::uniform_int_distribution<uint32_t> distribution(min, max);
    return distribution(generator);
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


void genData(std::string filename)
{
    std::ofstream file1(filename + "-in.csv");
    std::ofstream file2(filename + "-out.csv");
    for (uint32_t r = 0; r < 5000; r++) {
        float x = rand() / float(RAND_MAX);
        float y = rand() / float(RAND_MAX);
        file1 << x << ", " << y << std::endl;
        file2 << 2 * x + 10 + y << std::endl;
    }
    file1.close();
    file2.close();
}

std::vector<std::vector<float>> sliceData(std::string inputFile, float amount) 
{
    std::vector<std::vector<float>> target= readCSVToVector(inputFile);
    size_t n = round(target.size() * amount);
    std::vector<std::vector<float>> targetSlice(target.begin(), target.begin() + n);
    return targetSlice;
}



int main()
{   
    genData("test");

	std::vector<uint32_t> topology = {2,3,1};
	SimpleNeuralNetwork nn(topology, 0.1);

    float train = 0.9;
    float validate = 0.1; 
       
    std::vector<std::vector<float>> targetInputsTrain = sliceData("test-in.csv", train);
    std::vector<std::vector<float>> targetInputsValidate = sliceData("test-in.csv", validate);
    std::vector<std::vector<float>> targetOutputsTrain = sliceData("test-out.csv", train);
    std::vector<std::vector<float>> targetOutputsValidate = sliceData("test-out.csv", validate);
	
	uint32_t  epoch = 10000;
    uint32_t  k = 0;

    std::cout << "Population evaluation\n";

    uint32_t index = 100; // datos con los que vamos a evaluar la matriz
    
    for (uint32_t i = 0; i < nn._populationSize; i++)
    {
        nn.Initialize();
        float averError = 0;
        
        for (size_t j = 0; j < index; j++)
        {
            nn.FeedFordward(targetInputsTrain[j]);

            //  no backpropagation, vamos a ver cómo funciona esta inicializacion para 
            //  varios valores así como está

            auto preds = nn.getPrediction();
            std::vector<float> real = targetOutputsTrain[j];
            // La predicción sólo tiene un valor
            float errorP = abs(preds[0] - real[0]) * 100.0 / real[0];
            averError += errorP;
        }

        averError /= index; //sacamos en error promedio de la matriz para los primeros 100 datos
            
        _candidate Candidate;
        Candidate.weightC = nn._weightMatrices;
        Candidate.biasC = nn._biasMatrices;
        Candidate.error = averError;
        nn._population.push_back(Candidate);
        nn.flush();
        std::cout << "\rprogress : ";
        std::cout << static_cast<double>(i) * 100.0 / static_cast<double>(nn._populationSize) << " %" << std::flush;
    }
    std::cout << "Population evaluated\n";

    std::cout << "training started\n";

	for (uint32_t m = 0; m < epoch; m++)
	{
        nn.Crossover();
        float parError = 0;

        for (uint32_t i = 0; i < index; i++)
        {
            uint32_t p = getRandomNumber(0, targetInputsTrain.size() - 1);
            std::vector<float> t = targetInputsTrain[p];

            nn.FeedFordward(t);

            std::vector<float> real = targetOutputsTrain[p];
            auto preds = nn.getPrediction();
            parError += abs(preds[0] - real[0]) * 100.0 / real[0];
        }
        parError /= index;
        nn.Replace(parError);

        std::cout << "\rprogress : ";
        std::cout << static_cast<double>(m) * 100.0 / static_cast<double>(epoch) << " %" << std::flush;
	}

	std::cout << "training completed\n";

	for (auto input : targetInputsValidate)
	{
		nn.FeedFordward(input);
		auto preds = nn.getPrediction();
        for(uint32_t i=0; i < input.size()-1; i++){
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