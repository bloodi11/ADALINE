#pragma once
#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include <ctime>


class Neuron {
private:
	double learningRate;
	double bias;
	std::string activationFunction;  // "none" , "step", "relu" , "sigmoid"
	std::vector<double> inputData;
	std::vector<double> previousInputData;
	std::vector<double> weightData;	

	void tuneBias();
	void initWeightData();

	double sumBlock();
	double activationBlock();

	double stepFunction(double x);
	double reLUFunction(double x);
	double sigmoidFunction(double x);



public:
	Neuron();
	Neuron(unsigned int _size, double _learningRate, std::string _activationFunction);

	void setVectorInput(std::vector<double> _data);
    void setInput(const int _index, const double _data);
	double getInput(const int _index);

	void setWeight(const int _index, const double _data);
	double getWeight(const int _index);

	void setBias(const double _bias);

	void setInputSize(unsigned int _size);
	void setLearningRate(double _learningRate);

	int tuneWeights(int _iterations, double _desire);
	double calculateOutput();

	double calculateError(const double _desire);

};




