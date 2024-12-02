// this is a c++ class for constructing a Layer in a deep neural network 
// Author: Jakob Hoffmann 
#ifndef LAYERS_HPP
#define LAYERS_HPP

#include<iostream>
#include<vector>
#include "Tensor.hpp"
#include "Activation.hpp"



// class Layer 



template<class T>
class Layer
{

    

    template<class U>
    friend class Fullyconnected;
    template<class U>
    friend class Optimizer;
    public: 
    
    Tensor<T,2> weights;//define, but only initialize in constructor
    Tensor<T,1> bias;

    Tensor<T,1>z;// z=Wx+b usefull for backpropagation(else it needs to be calculated again)
    private:
    
    uint num_of_neurons;
    std::string activation;
    uint Layer_num; // serves as ID for layer

    
    public:
    std::vector<T (*)(Tensor<T, 1>)> free_funcs;
    Tensor<T,1> input;
    Tensor<T,1> output;
    
    Layer(int num_input, int num_output, std::string activation,int Layer_num) :
    activation(activation)
    ,input(std::vector<T>(num_input,0)),output(std::vector<T>(num_output,0))
    ,weights(std::vector<std::vector<T>>(num_output, std::vector<T>(num_input, 0)))
    ,bias(std::vector<T>(num_output,0)),z(std::vector<T>(num_output,0))
    {
        
        if(num_input<=0 || num_output<=0)
        {
            throw std::runtime_error("Negative dimension not allowed");
        }
        num_of_neurons=num_input;
    }

    Layer()=default;



    public:

    // initialize neurons

    void forward();// gives output of layer

    void free_forward();// gives output of layer

    // TODO: define function for layers 
    void backward(const Tensor<T,1>& ybatch);

    void initialize_weights(Tensor<T,2> weights, Tensor<T,1> bias)
    {
        this->weights=weights;
        this->bias=bias;
    }

    void initialize_weights(std::string mode="random_even")
    {
        if(mode=="random_even")
        {
            for(auto i=0;i<weights.getndims()[0];i++)
            {
                for(auto j=0;j<weights.getndims()[1];j++)
                {
                    weights(i,j)=double(rand())/RAND_MAX;
                }
            }
            for(auto i=0;i<bias.getsize();i++)
            {
                bias(i)=double(rand())/RAND_MAX;
            }
        }
        else if(mode=="zero")
        {
            for(auto i=0;i<weights.getndims()[0];i++)
            {
                for(auto j=0;j<weights.getndims()[1];j++)
                {
                    weights(i,j)=0.0;
                }
            }
            for(auto i=0;i<bias.getsize();i++)
            {
                bias(i)=0.0;
            }
        }
        
        else if(mode=="identity")
        {
            for(auto i=0;i<weights.getndims()[0];i++)
            {
                for(auto j=0;j<weights.getndims()[1];j++)
                {
                    if(i==j)
                    {
                        weights(i,j)=1.0;
                    }
                    else
                    {
                        weights(i,j)=0.0;
                    }
                }
            }
            for(auto i=0;i<bias.getsize();i++)
            {
                bias(i)=0.0;
            }
        }
        
        else
        {
            throw std::runtime_error("Mode not known");
        }
    }
    


};

// fully connected neural network

// 
template<class T>
class Convolutional : public Layer<T>
{

    // For later purposes; definition of convolutional neural network 

};

template<class T>
void Layer<T>::forward()
{

    //std::cout << weights << std::endl;
    //std::cout<<"Checkpoint11"<<std::endl;
    Tensor<T,1> temp = Matrix_vec_prod(weights,input);// contract<T,2,1,1>(weights,input,1,0);
   // std::cout << "W = " << weights << std::endl;
    //std::cout << "L = " << input << std::endl;
    //std::cout << "W*L = " << temp << std::endl;
    //std::cout << std::endl;
    
   // std::cout<<"Checkpoint12"<<std::endl;

    // std::cout<<"temp:"<<" "<<temp<<" "<<bias<<std::endl;

     output = temp+bias;

   

     //std::cout<<"Checkpoint13"<<std::endl;
    
    // apply activation function
    z=output;// save z for backpropagation

    // std::cout<<"Checkpoint14"<<std::endl;
    output=Activation(activation,output);

     //std::cout<<"Checkpoint15"<<std::endl;
    //std::cout<<output<<std::endl;
    
}

template<class T>
void Layer<T>::free_forward()
{
    Tensor<T,1> temp = Matrix_vec_prod(weights,input);// contract<T,2,1,1>(weights,input,1,0);
    z=temp+bias;
    //std::cout << "z: " << z << std::endl;
    //std::cout << "free_funcs.size(): " << free_funcs.size() << std::endl;
    for(auto i=0;i<z.getsize();i++)
    {
      //  std::cout << "z[" << i << "]: " << z[i] << std::endl;
        output[i]=free_funcs[i](z);
        //std::cout << "output[" << i << "]: " << output[i] << std::endl;
    }
    //std::cout << "output: " << output << std::endl;
}

#endif
