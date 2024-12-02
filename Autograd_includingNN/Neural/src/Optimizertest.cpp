#include "../lib/Optimizer.hpp"

#include <tuple>
#include<fstream>
#include<ctime>

// implementation of class networktest 

const int num_layers=6;
//std::vector<int> dim_layers{10,20,20,10};
//std::vector<int>dim_layers(num_layers,2);
std::vector<int>dim_layers = [] {
    std::vector<int> vec(num_layers, 100); // Initialize with all zeros
    vec[0] = 1;                // Set the first element to 1
    return vec;
}();
int dim_output=1;
std::vector<std::string> activations(num_layers,"tanh");
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
                ID_weights(i,j)=1.0;
            }
            else
            {
                ID_weights(i,j)=0.0;
            }
        }
        Zero_bias(i)=0;
    }
    return std::make_tuple(ID_weights,Zero_bias);


}

float f(float x)
{
    //return std::sqrt(x);
    //return 3+4*x;
    return sin(x);
}

int main()
{
    for(auto i=0;i<num_layers;i++)
    {
        auto result=initialize_Id_weights_zero_bias<float>(i,dim_layers,dim_output);
        Tensor<float,2> ID_weights=std::get<0>(result);
        Tensor<float,1> Zero_bias=std::get<1>(result);

        network.layers[i].initialize_weights(ID_weights,Zero_bias);

        Tensor<float,2> temp=ID_weights;

        //std::cout<<"Layer:"<<" "<<i<<" "<<"weights"<<" "<<temp<<std::endl;
    }

    /*
    std::vector<float> input(dim_layers[0],0.5);
    std::vector<int> dim = {2};
    Tensor<float,1> input_tensor(input);
    network.forward(input_tensor);
    
    std::cout << "For input: " << input_tensor << std::endl;
    std::cout << "Output: " << network.layers[num_layers-1].output << std::endl;
    */

    
    //std::vector<float> training_input{1,2,3,4};
    //std::vector<float> training_output{10,20,30,40};



    /*std::vector<Training_data<float>> td;

    std::vector<float> input{0.5,1.0,1.5,2.0,2.5,3.0,3.5,5.0,6.0,8.0};

    std::vector<float> output{1.1,1.9,3.0,5.5,4.8,6.1,6.8,10.5,12.0,15.5};


    Training_data<float> training_data(dim_layers[0],dim_output);

      Tensor<float,1> input_tensor_training(input);
      Tensor<float,1> output_tensor_training(output);
    
       training_data.input=input_tensor_training;
       training_data.target=output_tensor_training;

        td.push_back(training_data);*/

     std::vector<Training_data<float>> td;


    for(auto i=0;i<100;i++)
    {
       float training_input=0.0+10.0/100.0*i;
       float  training_output=f(training_input);

        Training_data<float> training_data(dim_layers[0],dim_output);

        Tensor<float,1> input_tensor_training(std::vector<float>{training_input});
        Tensor<float,1> output_tensor_training(std::vector<float>{training_output});
        
        training_data.input=input_tensor_training;
        training_data.target=output_tensor_training;

        td.push_back(training_data);
   

    }

     std::vector<Training_data<float>> testd;


    for(auto i=0;i<100;i++)
    {
       float  training_input=0.0+10.0/1000.0*i;
       float  training_output=f(training_input);

        Training_data<float> training_data(dim_layers[0],dim_output);

        Tensor<float,1> input_tensor_training(std::vector<float>{training_input});
        Tensor<float,1> output_tensor_training(std::vector<float>{training_output});
        
        training_data.input=input_tensor_training;
        training_data.target=output_tensor_training;

        testd.push_back(training_data);
   

    }
   

    
    
    
    

    Optimizer<float> optimizer(network,td,"square");
     int it=0;
    for(auto i=0;i<100;i++)
    {
        std::cout<<"step:"<<" "<<i<<std::endl;

         //std::cout<<"input"<<" "<<td[i%td.size()].input<<" "<<td[i%td.size()].target<<std::endl;
        optimizer.optimization_step(td[i%td.size()]);
       
        //optimizer.update_weights_and_biases_gradient_descent(0.001);
        optimizer.update_weights_and_biases_adam(0.01,0.9,0.999,1e-8,it);
       // std::cout << "Weights after training step: " << i << std::endl;
       //if(i%100==0 || i<10)
        std::cout <<"step"<<" "<<i<<" Loss:"<<" "<< square(network.layers[num_layers-1].output,td[0].target)<<std::endl;
        /*for(auto j=0;j<num_layers;j++)
        {
            std::cout << "Layer: " << j << std::endl;
            std::cout << "Weights: "<< network.layers[j].weights << std::endl;
            std::cout << "Bias: " << network.layers[j].bias << std::endl;
        }*/

        //std::cout << "Output after training step: " << i << std::endl;
        //network.forward(training_data.input);
        //std::cout << "Output: " << network.layers[num_layers-1].output << std::endl;
    }
        std::cout << "Output after all training step: "<< std::endl;
        //network.forward(td[0].input);
        std::cout << "Output: " << network.layers[num_layers-1].output << std::endl;

        std::ofstream file("train.dat");
        std::ofstream file2("func.dat");


        //Tensor<float,1> Test(std::vector<float>{0.3,5.4,2.4,1.2});

        for(auto i=0;i<td.size();i++)
        {
            network.forward(td[i].input);
           // network.forward(Test[i]);
            file<<td[i].input(0)<<" "<<network.layers[num_layers-1].output(0)<<std::endl;
            file2<<td[i].input(0)<<" "<<td[i].target(0)<<std::endl;

        }

        file.close();
  
        file2.close();

       
       std::ofstream file3("test.dat");
       

        for(auto i=0;i<testd.size();i++)
        {
            network.forward(testd[i].input);
           // network.forward(Test[i]);
            file3<<testd[i].input(0)<<" "<<network.layers[num_layers-1].output(0)<<std::endl;


        }

        file3.close();

       



    return 0;
};
