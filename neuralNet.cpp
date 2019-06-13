//modified and fixed version of tutorial code https://www.youtube.com/watch?v=m2QiLyVeiMM

#include "neuralNet.hpp"
#include <fstream>
#include <time.h>

NeuralNet::NeuralNet()
{
	topology.resize(0);
}

NeuralNet::NeuralNet(vector<unsigned int> layerSizes) : topology(move(layerSizes))
{
	srand(time(0));

	int numLayers = topology.size();

	weights.reserve(numLayers-1);

	for(int layerIndex = 1; layerIndex < numLayers; layerIndex++)
	{
		int currentSize = topology[layerIndex];
		int prevSize = topology[layerIndex-1];

		vector<vector<double>> layer(currentSize);

		auto gen = [prevSize]() {
			vector<double> result(prevSize+1);

			generate(result.begin(), result.end(), [](){
				return (static_cast<double>(rand())/RAND_MAX)*2. - 1.;
			});

			return result;
		};

		generate(layer.begin(), layer.end(), gen);

		weights.push_back(layer);
	}
}

NeuralNet::NeuralNet(vector<unsigned int> layerSizes, const double max) : topology(move(layerSizes))
{
	srand(time(0));

	int numLayers = topology.size();

	weights.reserve(numLayers-1);

	for(int layerIndex = 1; layerIndex < numLayers; layerIndex++)
	{
		int currentSize = topology[layerIndex];
		int prevSize = topology[layerIndex-1];

		vector<vector<double>> layer(currentSize);

		auto gen = [prevSize, max]() {
			vector<double> result(prevSize+1);

			generate(result.begin(), result.end(), [max](){
				return (static_cast<double>(rand())/RAND_MAX)*max - max/2.;
			});

			return result;
		};

		generate(layer.begin(), layer.end(), gen);

		weights.push_back(layer);
	}
}

double sigmoid(double val)
{
	return (1.0/(1.0+exp(-val)));
}

vector<double> NeuralNet::feedforward(const vector<double>& inputs, int index)
{
	vector<double> currentInput(inputs);
	currentInput.push_back(1);

	int numLayers = topology.size();

	for (int i = 1; i < numLayers && i <= index; i++)
	{
		vector<vector<double>>& layer = weights[i-1];

		int numNeurons = topology[i];
		vector<double> output(numNeurons);

		auto neuronVal = [&currentInput](vector<double>& neuronWeights){
			double value = inner_product(neuronWeights.begin(), neuronWeights.end(), currentInput.begin(), 0.0);
			double result = sigmoid(value);
			return result;
		};

		transform(layer.begin(), layer.end(), output.begin(), neuronVal);
		currentInput = move(output);

		if(i < numLayers - 1)
			currentInput.push_back(1);
	}
	return currentInput;
}

vector<double> NeuralNet::feedforward(const vector<double>& inputs)
{
	vector<double> currentInput(inputs);
	currentInput.push_back(1);

	int numLayers = topology.size();

	for (int i = 1; i < numLayers; i++)
	{
		vector<vector<double>>& layer = weights[i-1];

		int numNeurons = topology[i];
		vector<double> output(numNeurons);

		auto neuronVal = [&currentInput](vector<double>& neuronWeights){
			double value = inner_product(neuronWeights.begin(), neuronWeights.end(), currentInput.begin(), 0.0);
			double result = sigmoid(value);
			return result;
		};

		transform(layer.begin(), layer.end(), output.begin(), neuronVal);
		currentInput = move(output);

		if(i < numLayers - 1)
			currentInput.push_back(1);
	}
	return currentInput;
}

void NeuralNet::train(vector<vector<double>>& dset, unsigned int outputStart, double eta, double goalmse, int epochs)
{
	double mse = 1.;
	int epoch = -1;

	while(mse > goalmse && epoch < epochs)
	{
		epoch++;
		cout << "training epoch " << epoch << endl;
		vector<vector<double>> dataset = dset;
		random_shuffle(dataset.begin(), dataset.end());
		if(dataset.size() > 7500)
			dataset.resize(7500);


		for(unsigned int n = 0; n < dataset.size(); n++)
		{
			vector<double>& instance = dataset[n];
			vector<double> currentInput(instance.begin(),instance.begin()+outputStart);
			vector<double> expected(instance.begin()+outputStart,instance.end());
			
			vector<double> output(feedforward(currentInput));
			
			//forward
			vector<vector<double>> inputPerLayer;
			for(unsigned int layerIndex = 1; layerIndex < topology.size(); layerIndex++)
			{
				inputPerLayer.push_back(currentInput);
				currentInput = feedforward(currentInput, layerIndex);
			}

			//backward
			vector<double> error(expected);
			transform(error.begin(), error.end(), output.begin(), error.begin(), minus<double>());

			//sum squared error
			double tmperr = 0;
			for (unsigned int i = 0; i < error.size(); ++i)
			{
				error[i] = -error[i];
				tmperr += error[i];
			}
			mse += tmperr*tmperr;
			
			vector<double> sigma(error);
			transform(sigma.begin(), sigma.end(), output.begin(), sigma.begin(), [](double err, double actual){
				double delta = actual*(1-actual);
				return (err)*delta;
			});

			for(unsigned int layerIndex = topology.size()-1; layerIndex > 0; layerIndex--)
			{
				vector<vector<double>>& layer = weights[layerIndex-1];
				vector<double>& input = inputPerLayer[layerIndex-1];

				int prevNeurons = topology[layerIndex-1];
				int numNeurons = topology[layerIndex];

				vector<double> newSigma;
				for(int j = 0; j < prevNeurons; j++)
				{
					double newSigmaVal = 0.0;

					for(int s = 0, size = sigma.size(); s < size; s++)
					{
						newSigmaVal = newSigmaVal + sigma[s]*layer[s][j];
					}
					newSigmaVal = newSigmaVal*input[j]*(1-input[j]);

					newSigma.push_back(newSigmaVal);
				}

				//update
				for(int j = 0; j < numNeurons; j++)
				{
					vector<double>& neuronWeight = layer[j];
					double factor = eta * sigma[j];

					transform(neuronWeight.begin(), neuronWeight.end(), input.begin(), neuronWeight.begin(), [factor](double oldWeight, double inputVal) {
						return oldWeight - factor * inputVal;
					});
				}

				sigma = move(newSigma);
			}
		}

		mse = mse/(double)dataset.size();
		cout << "MSE: " << mse << endl;
	}
}


ostream& operator<<(ostream& os, const NeuralNet& net)
{
	os << setprecision(numeric_limits<long double>::digits10 + 1);
	os << net.topology.size() << endl;
	for (unsigned int i = 0; i < net.topology.size(); ++i)
	{
		os << net.topology[i] << " ";
	}
	os << endl;

	for(unsigned int i = 0; i < net.weights.size(); i++)
	{
		for(unsigned int j = 0; j < net.weights[i].size(); j++)
		{
			for(unsigned int k = 0; k < net.weights[i][j].size(); k++)
			{
				os << net.weights[i][j][k] << " ";
			}
			os << endl;
		}
		os << endl;
	}
	return os;
}

istream& operator>>(istream& is, NeuralNet& net)
{
	is >> setprecision(numeric_limits<long double>::digits10 + 1);
	unsigned int size;
	is >> size;
	net.topology.resize(size);
	for (unsigned int i = 0; i < size; ++i)
	{
		is >> net.topology[i];
	}
	is.ignore(100,'\n');

	net = NeuralNet(net.topology);

	for(unsigned int i = 0; i < net.weights.size(); i++)
	{
		for(unsigned int j = 0; j < net.weights[i].size(); j++)
		{
			for(unsigned int k = 0; k < net.weights[i][j].size(); k++)
			{
				is >> net.weights[i][j][k];
			}
			is.ignore(100,'\n');
		}
		is.ignore(100,'\n');
	}
	return is;
} 