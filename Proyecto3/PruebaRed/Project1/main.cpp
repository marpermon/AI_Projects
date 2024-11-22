#include <iostream>
#include "NeuralNetwork.h"


int main()
{
	std::vector<uint32_t> topology = { 2,3,1 };
	SimpleNeuralNetwork nn(topology, 0.1);

	std::vector<std::vector<float>> targetInputs{
		{0.0f,0.0f},
		{1.0f,1.0f},
		{1.0f,0.0f},
		{0.0f,1.0f}
	};

	std::vector<std::vector<float>> targetOutputs{
		{0.0f},
		{0.0f},
		{1.0f},
		{1.0f}
	};

	uint32_t  epoch = 100000;

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