#include <iostream>
#include "neuralNet.hpp"
#include <fstream>
#include <iomanip>
#include <numeric>

using namespace std;

int main(int argc, char const *argv[])
{
	cout << setprecision(numeric_limits<long double>::digits10 + 1);
	//gen new network
	if(argc > 1 && (string)argv[1] == "gen")
	{
		NeuralNet net({2,2,1}, 0.5);
		ofstream fout("network.txt");
		fout << net << endl;
		fout.close();
		return 0;
	}

	double eta = 0.5;
	double mse = 0.000001;
	// int epochs = 1000000; 

	//train new network
	{
		//load network
		NeuralNet net;
		ifstream fin("network.txt");
		fin >> net;
		fin.close();

		//load dataset
		vector<vector <double>> dataset = {
			{0,0,0},
			{1,0,1},
			{0,1,1},
			{1,1,0}
		};

		cout << "starting results" << endl;
		for (unsigned int i = 0; i < dataset.size(); ++i)
		{
			vector<double> input = {dataset[i][0], dataset[i][1]};
			cout << input[0] << " " << input[1] << endl;
			cout << net.feedforward(input)[0] << endl;
		}

		//train network
		cout << "training" << endl;
		net.train(dataset, net.inputs(), eta, mse);//, epochs);
		cout << "done training" << endl;

		// output = net.feedforward(tmpinput);

		cout << "final results" << endl;
		for (unsigned int i = 0; i < dataset.size(); ++i)
		{
			vector<double> input = {dataset[i][0], dataset[i][1]};
			cout << input[0] << " " << input[1] << endl;
			cout << net.feedforward(input)[0] << endl;
		}
		
		//save network
		ofstream fout("newnetwork.txt");
		fout << net;
		fout.close();
	}

	return 0;
}