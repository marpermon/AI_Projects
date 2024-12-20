#pragma once
#include "Matrix.h"
#include <cstdlib>
#include <algorithm>
#include <cmath>

/*

[] -> []
[] -> []


*/

inline float Activation(float x) 
{
	return (x > 0) ? x : 0; //ReLU
}

inline float DActivation(float x)
{
	return (x > 0) ? 1.0 : 0.0;
}

struct _candidate
{
	std::vector<Matrix> weightC;
	std::vector<Matrix> biasC;
	float error;
};

class SimpleNeuralNetwork
{
public:
	std::vector<uint32_t> _topology;
	std::vector<Matrix> _weightMatrices;
	std::vector<Matrix> _valueMatrices;
	std::vector<Matrix> _biasMatrices;
	float _learningRate;
	std::vector<_candidate> _population;
	uint32_t _populationSize;

public:
	SimpleNeuralNetwork(std::vector<uint32_t> topology, float
		learningRate = 0.1f, uint32_t populationSize = 50)
		:_topology(topology), 
		_weightMatrices({}),
		_valueMatrices({}),
		_biasMatrices({}),
		_learningRate(learningRate),
		_populationSize(populationSize){}

	void Initialize(){
		for (uint32_t i = 0; i < _topology.size() - 1; i++)  //Aquí es donde se utiliza la topología, para crear una cantidad n de matrices de pesos
		{
			Matrix weightMatrix(_topology[i + 1], _topology[i]);
			weightMatrix = weightMatrix.applyFunction(
				[](const float& f) {
					return (float)rand() / RAND_MAX;
				}
			);
			_weightMatrices.push_back(weightMatrix);
			Matrix biasMatrix(_topology[i + 1], 1);
			biasMatrix = biasMatrix.applyFunction(
				[](const float& f) {
					return ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
				}
			);
			_biasMatrices.push_back(biasMatrix);
		}
		_valueMatrices.resize(_topology.size());
	}


	void flush() {
		_weightMatrices = {};
		_valueMatrices = {};
		_biasMatrices = {};
	}

	bool FeedFordward(std::vector<float> input)
	{
		if (input.size() != _topology[0])
			return false;
		Matrix values(input.size(), 1); 	//dudas con esto. debería el vector cambiar de tamaño?
		for (size_t i = 0; i < input.size(); i++)
			values._vals[i] = input[i];
		
		for (size_t  i = 0; i < _weightMatrices.size(); i++)
		{
			_valueMatrices[i] = values;
			values = values.multiply(_weightMatrices[i]);
			values = values.add(_biasMatrices[i]);
			values = values.applyFunction(Activation);
		}
		_valueMatrices[_weightMatrices.size()] = values;
		return true;
	}

	bool FeedFordward2(std::vector<float> input)
	{
		if (input.size() != _topology[0])
			return false;
		Matrix values(input.size(), 1);//dudas con esto. debería el vector cambiar de tamaño?
		for (size_t i = 0; i < input.size(); i++)
		{
			values._vals[i] = input[i];
			std::cout << values._vals[i] << "\n";
		}
		
		for (size_t i = 0; i < _weightMatrices.size(); i++)
		{
			_valueMatrices[i] = values;
			values = values.multiply(_weightMatrices[i]);
			values = values.add(_biasMatrices[i]);
			values = values.applyFunction(Activation);
		}
		
		_valueMatrices[_weightMatrices.size()] = values;
		return true;
	}

	// Function to sort the vector of structs
	void sortCandidates() {
		std::sort(_population.begin(), _population.end(), [](const _candidate& a, const _candidate& b) {
			return a.error< b.error; // Sort in ascending order
			}); 
	}


	bool Crossover()
	{
		_valueMatrices.resize(_topology.size());
		sortCandidates();
		uint32_t parent1 = rand() % (int)(_populationSize / 2.0);
		uint32_t parent2 = rand() % (int)(_populationSize / 2.0);
		std::vector<Matrix> OffWeights = {};
		std::vector<Matrix> OffBiases = {};
		
		for (size_t i = 0; i < _topology.size() - 1; i++)
		{
			Matrix OffspringWeight = _population[parent1].weightC[i].averageElements( _population[parent2].weightC[i] );
			Matrix OffspringBias = _population[parent1].biasC[i].averageElements( _population[parent2].biasC[i] );
			OffWeights.push_back(OffspringWeight);
			OffBiases.push_back(OffspringBias);
		}

		_weightMatrices = OffWeights;
		_biasMatrices = OffBiases;

		return true;
	}


	void Replace(float error) {
		if (error < _population.back().error) {
			_candidate NewCandidate;
			NewCandidate.weightC = _weightMatrices;
			NewCandidate.biasC = _biasMatrices;
			NewCandidate.error = error;
			_population.back() = NewCandidate;
			sortCandidates();
		}
	}

	std::vector<float> getPrediction()
	{
		return _valueMatrices.back()._vals;
	}
};
