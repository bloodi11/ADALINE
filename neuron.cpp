#include "stdafx.h"

Neuron::Neuron() : Neuron(1, 1, "none") {
	setWeight(0, 1);
	bias = 0;
}

Neuron::Neuron(unsigned int _size, double _learningRate, std::string _activationFunction) : learningRate(_learningRate), activationFunction(_activationFunction) {
	inputData.resize(_size);
	previousInputData.resize(_size);
	previousInputData[0] = -2;
	weightData.resize(_size);
	initWeightData();
	tuneBias();
}


void Neuron::tuneBias() {
	bias = 0 - (((double)inputData.size() ) / (10 * learningRate + 0.01));
	learningRate = std::abs(learningRate);
}

void Neuron::setBias(double _bias) {
	bias = _bias;
}

void Neuron::setVectorInput(std::vector<double> _data) {
	inputData = _data;
}

void Neuron::setInput(const int _index, const double _data) {
	inputData[_index] = _data;
}

double Neuron::getInput(const int _index) {
	return inputData[_index];
}

void Neuron::setWeight(const int _index, const double _data) {
	weightData[_index] = _data;
}


double Neuron::getWeight(const int _index) {
	return weightData[_index];
}

int Neuron::tuneWeights(int _iterations, double _desire) {
	std::vector<int> indexes;
	for (unsigned int i = 0; i <= weightData.size() - 1; i++) {
		indexes.push_back(i);
	}
	//std::vector<double> previousW;
	//previousW = weightData;

	for (int i = 1; i <= _iterations; i++) {
		std::random_shuffle(indexes.begin(), indexes.end());
		for (auto& j : indexes) {
			//double momentum = weightData[j] - previousW[j];
			//previousW[j] = weightData[j];
			if (inputData[j] == previousInputData[j] || previousInputData[0] == -2) {
				weightData[j] += learningRate*(_desire - calculateOutput())*inputData[j];
			}
			if (inputData[j] != previousInputData[j] && previousInputData[0] != -2) {
				weightData[j] = 0;
			}
			if (calculateError(_desire) < 1e-9) {
				previousInputData = inputData;
				std::cout << "iterations: " << i << std::endl;
				return 0;
			}
		}
	}
	previousInputData = inputData;
	std::cout << "iterations: " << "all possible" << std::endl;

	return 0;
}


double Neuron::calculateOutput() {
	double result = activationBlock();
	return result;
}

double Neuron::calculateError(const double _desire) {
		double error = std::pow(_desire - calculateOutput(), 2);
		double result = error / 2;

	return result;
}

void Neuron::initWeightData() {
	std::srand(std::time(nullptr));
	for (unsigned int i = 0; i <= inputData.size() - 1; i++) {
		weightData[i] =  ( (double)i ) / ( std::rand() ) ;
	}
}

double Neuron::sumBlock() {
	int i = 0;
	double membranePotential = 0.0;
	for (auto& input : inputData) {
		membranePotential += input * weightData[i];
		i++;
	}
	membranePotential += bias;
	return membranePotential;
}

double Neuron::activationBlock() {
	if (activationFunction == "none") {
		double result = sumBlock();
		return result;
	}
	else if (activationFunction == "step") {
		double result = stepFunction(sumBlock());
		return result;
	}
	else if (activationFunction == "relu") {
		double result = reLUFunction(sumBlock());
		return result;
	}
	else if (activationFunction == "sigmoid") {
		double result = sigmoidFunction(sumBlock());
		return result;
	}
	else
		return sumBlock();
}

double Neuron::stepFunction(double x) {
	double result = sumBlock();
	if (result > 0) return 1;
	else
		return 0;
}

double Neuron::reLUFunction(double x) {
	double result = std::max(0.0, x);
	return result;
}

double Neuron::sigmoidFunction(double x) {
	double result = 1 / (1 + std::exp(-x));
	return result;
}

void Neuron::setInputSize(unsigned int _size) {
	inputData.resize(_size);
	previousInputData.resize(_size);
	weightData.resize(_size);
}

void Neuron::setLearningRate(double _learningRate) {
	learningRate = _learningRate;
	tuneBias();
}


