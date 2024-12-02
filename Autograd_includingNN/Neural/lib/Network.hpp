// in this c++ class the deep learing model is generated.
// Author: Jkaob Hoffmann 

#ifndef NETWORK_HPP
#define NETWORK_HPP
#include "Layers.hpp"
#include<vector>   
#include "Loss.hpp"



template<class T>
class Fullyconnected
{
    public: 
    std::vector<int> dim_layers;
    std::vector<Layer<T>> layers;
    const int output_size;
    //Layer<T>& output_layer; //special_name for layers[layers.size()-1]
    //Layer<T>& input_layer; //special_name for layers[0]
    std::string loss_function;

    void initialize_weights(std::string mode = "random_even")
    {
        for(auto i=0;i<layers.size();i++)
        {
            layers[i].initialize_weights(mode);
        }
    }

   //TODO: implement geting network from file
    Fullyconnected(int output_size,std::vector<int> dim_layers,std::vector<std::string> activations):output_size(output_size),dim_layers(dim_layers)
    
    {
        for(auto i=0;i<dim_layers.size()-1;i++)
        {
            Layer<T> layer(dim_layers[i],dim_layers[i+1],activations[i],i);
            layers.push_back(layer);
        }
        Layer<T> temp(dim_layers[dim_layers.size()-1],output_size,activations[activations.size()-1],dim_layers.size()-1);
        layers.push_back(temp);
    }
    
    //implement getting network from network
    
    void forward(Tensor<T,1> input)
    {
        layers[0].input=input;
        
        for(auto i=0;i<layers.size()-1;i++)
        {
            layers[i].forward();

            layers[i+1].input=layers[i].output;
        }
        
        int last=layers.size()-1;
    
        layers[layers.size()-1].forward();
    }

    void free_forward(Tensor<T,1> input)
    {
        
        layers[0].input=input;
        //std::cout << "Checkpoint 1" << std::endl;
        
        for(auto i=0;i<layers.size()-1;i++)
        {
            layers[i].free_forward();
            //std::cout << "\nLayer " << i << " output: " << layers[i].output << std::endl;
            layers[i+1].input=layers[i].output;
            //std::cout << "Layer " << i+1 << " input: " << layers[i+1].input << std::endl;
        }
        
        int last=layers.size()-1;
    
        layers[layers.size()-1].free_forward();
    }
    //void save_weights_and_biases(std::string filename);


};




#endif