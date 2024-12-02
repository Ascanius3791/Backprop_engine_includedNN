#include "../lib/Networktest.hpp"


#include <iostream>
#include <vector>
#include <tuple>
// implementation of class networktest 

int num_layers=7;
std::vector<int> dim_layers{2,2,8,8,8,4,8};
int dim_output=8;
std::vector<std::string> activations(num_layers,"ReLU");
Fullyconnected<float> network(dim_output,dim_layers,activations);


template<class T>
std::tuple<Tensor<float,2>,Tensor<float,1>> initialize_Id_weights_zero_bias(int layer_num,std::vector<int> dim_layers,int dim_output)
{
    int input_dim=dim_layers[layer_num];
    int output_dim;
    if(layer_num==dim_layers.size()-1)
    {
        output_dim=dim_output;
    }
    else
    {
        output_dim=dim_layers[layer_num+1];
    }
    
    Tensor<float,2> ID_weights(std::vector<int>{output_dim,input_dim});
    Tensor<float,1> Zero_bias(std::vector<int>{output_dim});
    for(auto i=0;i<output_dim;i++)
    {
        for(auto j=0;j<input_dim;j++)
        {
            if(i==j)
            {
                ID_weights(i,j)=1;
            }
            else
            {
                ID_weights(i,j)=0;
            }
        }
        Zero_bias(i)=0;
    }
    return std::make_tuple(ID_weights,Zero_bias);


}


int main()
{
    
    
    
    for(auto i=0;i<num_layers;i++)
    {
        auto result=initialize_Id_weights_zero_bias<float>(i,dim_layers,dim_output);
        Tensor<float,2> ID_weights=std::get<0>(result);
        Tensor<float,1> Zero_bias=std::get<1>(result);

        network.layers[i].initialize_weights(ID_weights,Zero_bias);
    }
    std::vector<float> input(dim_layers[0],0.5);
    std::vector<int> dim = {2};
    Tensor<float,1> input_tensor(input);
    
    
    network.forward(input_tensor);
    std::cout << "For input: " << input_tensor << std::endl;
    std::cout << "Output: " << network.layers[num_layers-1].output << std::endl;
    


    return 0;
};
