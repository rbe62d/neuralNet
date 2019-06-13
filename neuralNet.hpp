//modified version of tutorial code https://www.youtube.com/watch?v=m2QiLyVeiMM

#pragma once

#include <vector>
#include <algorithm>
#include <functional>
#include <cmath>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <limits>

using namespace std;

class NeuralNet
{
private:
	vector<unsigned int> topology;
	vector<vector<vector<double>>> weights;
public:
	vector<double> feedforward(const vector<double>& inputs, int index);
	NeuralNet();
	NeuralNet(vector<unsigned int> layerSizes);
	NeuralNet(vector<unsigned int> layerSizes, const double max);
	virtual ~NeuralNet() {};

	vector<double> feedforward(const vector<double>& inputs);

	void train(vector<vector<double>>& dataset, unsigned int outputStart, double eta, double goalmse, int epochs = 1000000);

	int inputs() {return topology[0];} 

	friend ostream& operator<<(ostream& os, const NeuralNet& net); 
	friend istream& operator>>(istream& is, NeuralNet& net); 
};

// #include "neuralNet.cpp"