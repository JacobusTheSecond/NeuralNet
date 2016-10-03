#include <cmath>
#include <string>
#include <iostream>
#include <sstream>
#include <time.h>

class Node
{
public:
	Node();
	Node(int psize, Node** p);
	Node(int psize, Node** p, double* pweight);
	~Node();
	double getParentWeightAt(int i);
	double getValue();
	void setParentWeightAt(int i, double v);
	Node* getParentAt(int i);
	bool setValue(double v);
private:
	Node** parent;
	int parentsize;
	double* parentweight;
	double value;
	bool superparent = true;
};


class NeuralNet
{
public:
	NeuralNet(int iLength, int oLength);
	NeuralNet(int iLength, int hLength, int oLength);
	double* simulate(double input[]);
	bool backprop(double input[], double expout[], double learnrate);
	~NeuralNet();
	std::string print();
private:
	Node** inputLayer;
	Node** hiddenLayer;
	Node** outputLayer;
	int inputLength;
	int hiddenLength;
	int outputLength;
};

NeuralNet::NeuralNet(int iLength, int oLength)
{
	NeuralNet(iLength, iLength, oLength);
}

NeuralNet::NeuralNet(int iLength, int hLength, int oLength)
{
	inputLength = iLength;
	hiddenLength = hLength;
	outputLength = oLength;
	inputLayer = new Node*[iLength];
	for (int i = 0; i < iLength; ++i) 
	{
		Node* inodeptr = new Node();
		inputLayer[i] = inodeptr;
	}

	hiddenLayer = new Node*[hiddenLength];
	for (int i = 0; i < hiddenLength; ++i)
	{
		Node* hnodeptr = new Node(inputLength, inputLayer);
		hiddenLayer[i] = hnodeptr;
	}

	outputLayer = new Node*[oLength];
	for (int i = 0; i < oLength; ++i) 
	{
		Node* onodeptr = new Node(hiddenLength, hiddenLayer);
		outputLayer[i] = onodeptr;
	}

}

double* NeuralNet::simulate(double input[])
{
	for (int i = 0; i<inputLength; ++i) {
		(*(inputLayer[i])).setValue(input[i]);
	}

	double* output = new double[outputLength];
	for (int i = 0; i < outputLength; ++i) {
		output[i] = (*(outputLayer[i])).getValue();
	}
	return output;
}

bool NeuralNet::backprop(double input[], double expout[], double learnrate)
{

	double* output;
	output  = simulate(input);

	//get modified weights for connections between hidden Layer and output Layer
	double** mWeight1 = new double *[outputLength];
	for (int i = 0; i < outputLength; i++)
	{
		mWeight1[i] = new double[hiddenLength];
		for (int j = 0; j < hiddenLength; ++j) 
		{
			double deltaoi = -(expout[i] - output[i])*output[i] * (1 - output[i]);
			double outhj = (outputLayer[i]->getParentAt(j))->getValue();
			mWeight1[i][j] = outputLayer[i]->getParentWeightAt(j) - learnrate*deltaoi*outhj;
		}
	}

	//get modified weights for connections between input layer and hidden layer
	double** mWeight2 = new double *[hiddenLength];
	for (int i = 0; i < hiddenLength; i++)
	{
		mWeight2[i] = new double[inputLength];
		for (int j = 0; j < inputLength; ++j) 
		{
			double dEtot = 0;
			for (int k = 0; k < outputLength; ++k) {
				dEtot += -(expout[k] - output[k]) *output[k] * (1 - output[k])*outputLayer[k]->getParentWeightAt(i);
			}

			double dhi = hiddenLayer[i]->getValue() * (1 - hiddenLayer[i]->getValue());
			mWeight2[i][j] = hiddenLayer[i]->getParentWeightAt(j) - learnrate*dEtot * dhi * input[j];
		}
	}

	//apply mWeight1 and mWeight2
	for (int i = 0; i < outputLength; ++i) {
		for (int j = 0; j < hiddenLength; ++j) {
			outputLayer[i]->setParentWeightAt(j, mWeight1[i][j]);
		}
	}

	for (int i = 0; i < hiddenLength; ++i) {
		for (int j = 0; j < inputLength; ++j) {
			hiddenLayer[i]->setParentWeightAt(j, mWeight2[i][j]);
		}
	}


	//delete shit
	for (int i = 0; i < outputLength; ++i) {
		delete mWeight1[i];
	}
	delete mWeight1;
	for (int i = 0; i < hiddenLength; ++i) {
		delete mWeight2[i];
	}
	delete mWeight2;
	return true;
}

NeuralNet::~NeuralNet()
{
	for (int i = 0; i < inputLength; ++i) {
		delete inputLayer[i];
	}
	delete inputLayer;
	for (int i = 0; i < hiddenLength; ++i) {
		delete hiddenLayer[i];
	}
	delete hiddenLayer;
	for (int i = 0; i < outputLength; ++i) {
		delete outputLayer[i];
	}
	delete outputLayer;


}

std::string NeuralNet::print()
{
	std::stringstream ss;
	for (int i = 0; i < hiddenLength; ++i) {
		for (int j = 0; j < inputLength; ++j) {
			ss << "i"<<j<<" has link to h"<<i<<" of weight:"<<hiddenLayer[i]->getParentWeightAt(j)<<std::endl;
		}
	}

	for (int i = 0; i < outputLength; ++i) {
		for (int j = 0; j < hiddenLength; ++j) {
			ss << "h" << j << " has link to o" << i << " of weight:" << outputLayer[i]->getParentWeightAt(j) << std::endl;
		}
	}
	return ss.str();
}

Node::Node()
{
}

Node::Node(int psize, Node ** p)
{
	parentsize = psize;
	parent = p;
	superparent = false;
	parentweight = new double[parentsize];
	for (int i = 0; i < parentsize; ++i) {
		parentweight[i] = ((double)rand() / (double)RAND_MAX);
	}
}

Node::Node(int psize, Node ** p, double * pweight)
{
	parentsize = psize;
	parent = p;
	superparent = false;
	parentweight = pweight;
}

Node::~Node()
{
	delete parentweight;
}

double Node::getParentWeightAt(int i)
{
	if (i >= parentsize) {
		return 0;
	}
	else {
		return parentweight[i];
	}
}

double Node::getValue()
{
	if (!superparent) {
		value = 0; 
		for (int i = 0; i < parentsize; ++i) {
			value += (*(parent[i])).getValue() * parentweight[i];
		}
		value = 1 / (1 + exp(-value));
	}

	return value;
}

void Node::setParentWeightAt(int i, double v)
{
	if (i < parentsize) 
	{
		parentweight[i] = v;
	}
}

Node * Node::getParentAt(int i)
{
	if (i >= parentsize) {
		return nullptr;
	}
	else {
		return parent[i];
	}
}

bool Node::setValue(double v)
{
	if (superparent) {
		value = v;
		return true;
	}
	else {
		return false;
	}
}

int main() {
	//test
	srand(time(NULL));
	NeuralNet nn (2,4,1);
	double array00[2] = { 0.0 , 0.0 };
	double array01[2] = { 0.0 , 1.0};
	double array02[2] = { 1.0 , 0.0  };
	double array03[2] = { 1.0 , 1.0};

	double array0[1] = { 0 };
	double array1[1] = { 1 };
	double array2[1] = { 0.5 };

	std::cout << "before: " << std::endl;

	double* arrayb = nn.simulate(array00);
	std::cout << "(" << array00[0] << "," << array00[1] << ") -> (" << arrayb[0] << "," << ")" << std::endl;
	arrayb = nn.simulate(array01);
	std::cout << "(" << array01[0] << "," << array01[1] << ") -> (" << arrayb[0] << ","  << ")" << std::endl;
	arrayb = nn.simulate(array02);
	std::cout << "(" << array02[0] << "," << array02[1] << ") -> (" << arrayb[0] << ","  << ")" << std::endl;
	arrayb = nn.simulate(array03);
	std::cout << "(" << array03[0] << "," << array03[1] << ") -> (" << arrayb[0] << ","  << ")" << std::endl;

	//std::cout << std::endl << nn.print() << std::endl << std::endl;

	for (int i = 0; i < 10000; ++i) {
		double array[2] = {((double)rand() / (double)RAND_MAX), ((double)rand() / (double)RAND_MAX) };
		double arrayrev[2] = { array[0],array[1] };
		nn.backprop(array00, array0, 13);
		nn.backprop(array01, array1, 13);
		nn.backprop(array02, array1, 13);
		nn.backprop(array03, array1, 13);
		
	}
	std::cout << "after: " << std::endl;
	
	//::cout << " modifiziert zu: " << std::endl << std::endl << nn.print() << std::endl << std::endl << std::endl;
	double* array = nn.simulate(array00);
	std::cout << "(" << array00[0] << "," << array00[1] << ") -> (" << array[0] << ","  << ")"<< std::endl;
	array = nn.simulate(array01);
	std::cout << "(" << array01[0] << "," << array01[1] << ") -> (" << array[0] << ","  << ")" << std::endl;
	array = nn.simulate(array02);
	std::cout << "(" << array02[0] << "," << array02[1] << ") -> (" << array[0] << ","  << ")" << std::endl;
	array = nn.simulate(array03);
	std::cout << "(" << array03[0] << "," << array03[1] << ") -> (" << array[0] << ","  << ")" << std::endl;
	system("pause");
	return 0;
}