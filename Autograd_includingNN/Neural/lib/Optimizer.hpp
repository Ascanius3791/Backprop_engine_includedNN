// class for backward propagation 
// Author : Jakob Hoffmann 
#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "../lib/Network.hpp"


template<class T>
struct Training_data
{
    Tensor<T,1> input;
    Tensor<T,1> target;
    Training_data(int input_size,int target_size):input(std::vector<T>(input_size,0)),target(std::vector<T>(target_size,0))
    {
        if(input_size<=0 || target_size<=0)
        {
            throw std::runtime_error("Negative dimension not allowed");
        }
    }
};

template<class T>
class Del_Weights_and_Bias //for a given layer (k)// gives nummerical Partial derivatives of the Loss function wrt. the layer entrys L^(k)_i the weights W^(k)_ij and the biases b^(k)_i. 
{
    public:
    Tensor<T,2> del_W;//                    dLoss/dW^(k)_ij
    Tensor<T,1> del_b;//                    dLoss/db^(k)_i
    Tensor<T,2> mw;  // only necessary for Adam 
    Tensor<T,2> vw;
    Tensor<T,1> mb;
    Tensor<T,1> vb;
    //del_Loss()=default;
    Del_Weights_and_Bias(Layer<T>& layer):del_W(layer.weights.getndims()),del_b(layer.bias.getndims()),mw(layer.weights.getndims()),vw(layer.weights.getndims()), mb(layer.bias.getndims()),vb(layer.bias.getndims())
    {
        for(auto i=0;i<del_W.getndims()[0];i++)
        {
            for(auto j=0;j<del_W.getndims()[1];j++)
            {
                del_W(i,j)=0.0;
                vw(i,j)=0.0;
                mw(i,j)=0.0;
            }
        }
        for(auto i=0;i<del_b.getsize();i++)
        {
            del_b(i)=0.0;
            mb(i)=0.0;
            vb(i)=0.0;
        }
    }

    void reset_Del_Weights_and_Bias();

   
    T get_max_abs_W_b()
{
    T max=0.0;
    for(auto i=0;i<del_W.getndims()[0];i++)
    {
        for(auto j=0;j<del_W.getndims()[1];j++)
        {
            if(std::abs(del_W(i,j))>max)
            {
                max=std::abs(del_W(i,j));
            }
        }
    }
    for(auto i=0;i<del_b.getsize();i++)
    {
        if(std::abs(del_b(i))>max)
        {
            max=std::abs(del_b(i));
        }
    }
    return max;
}

};



template<class T>
void Del_Weights_and_Bias<T>::reset_Del_Weights_and_Bias()
    {
        for(auto i=0;i<del_W.getndims()[0];i++)
        {
            for(auto j=0;j<del_W.getndims()[1];j++)
            {
                del_W(i,j)=0.0;
            }
        }
        for(auto i=0;i<del_b.getsize();i++)
        {
            del_b(i)=0.0;
        }
    }

template<class T> 
class Optimizer//based on training data optimizes the network. This is not safe for multithreading!!! you have to allocate multiple optimizers for multithreading
{
    private:
    Fullyconnected<T>& network;
    std::vector<Training_data<T>> training_data;
    std::vector<Del_Weights_and_Bias<T>> sum_changes_of_weights_and_biases;
    std::string Loss_function;

    Tensor<T,1> output;

    public:

    std::vector<Del_Weights_and_Bias<T>> get_sum_changes_of_weights_and_biases()
    {
        return sum_changes_of_weights_and_biases;
    }



    Optimizer(Fullyconnected<T>& network,std::vector<Training_data<T>> training_data,std::string Loss_func):network(network),training_data(training_data)
    {
        Loss_function=Loss_func;
        //std::cout << "Networ size: " << network.layers.size() << std::endl << std::endl;
        for(auto i=0;i<network.layers.size();i++)
        {   
            
            Del_Weights_and_Bias<T> temp(network.layers[i]);
            sum_changes_of_weights_and_biases.push_back(temp);
        }
        //std::cout << "Optimizer initialized" << std::endl << std::endl;
    }
    
    void reset_sum_changes_of_weights_and_biases()
    {
        for(auto i=0;i<sum_changes_of_weights_and_biases.size();i++)
        {
            sum_changes_of_weights_and_biases[i].reset_Del_Weights_and_Bias();
        }
    }

    void is_plausible_sum_changes_of_weights_and_biases(std::string message="")
    {
        T threshhold=1e3;
        for(auto i=0;i<sum_changes_of_weights_and_biases.size();i++)
        {
            if(sum_changes_of_weights_and_biases[i].get_max_abs_W_b()>threshhold)
            {
                std::cout << "Max abs W_b: " << sum_changes_of_weights_and_biases[i].get_max_abs_W_b() << std::endl;
                if(message!="")
                {
                    std::cout << message << std::endl;
                }
                throw std::runtime_error("Plausibility of W or b check failed");
            }
        }
    }

    void optimization_step(Training_data<T> training_data)
    {
        is_plausible_sum_changes_of_weights_and_biases("Before optimization step");
        //check if weights or biases are nan
        for(auto i=0;i<network.layers.size();i++)
        {
            network.layers[i].weights.isnan("None Weight detected, going into optimization step");
            network.layers[i].bias.isnan("None Bias detected, going into optimization step");
        }
        network.forward(training_data.input);//forward pass leaves z and outputs accessible


        // dL/dL^(k)_i
        std::vector<int> dim;//dimension of second index of W^(k)_ij
        dim.push_back(0);//this is just a filler! we need Dim(dL/dL^(k+1)) = Dim(z^(k))
        for(auto i=1;i<network.layers.size();i++)
        {
            dim.push_back(network.layers[i].num_of_neurons);
        }
        dim.push_back(network.output_size);
        
        std::vector<Tensor<T,1>> d_Loss_d_L;
        for(auto i=0;i<dim.size();i++)//the first entry is just a filler for easier arythmetics
        {      
            Tensor<T,1> temp(std::vector<T>(dim[i],0));
            d_Loss_d_L.push_back( temp);
        }
        for(auto i=0;i<dim.size();i++)
        {
            d_Loss_d_L[i].isnan();
            //std::cout << "d_Loss_d_L[" << i << "]: " << d_Loss_d_L[i] << std::endl;
        }
        
        Tensor<T,1> dLoss=D_Loss<T,T>(network.output_size,Loss_function,network.layers[network.layers.size()-1].output,training_data.target);
        //std::cout << "dLoss: " << dLoss << std::endl;
        d_Loss_d_L[network.layers.size()] = dLoss;
        /*for(auto i=0;i<dim.size()-1;i++)
        {
            //d_Loss_d_L[i].isnan();
            std::cout << "d_Loss_d_L[" << i+1 << "]: " << d_Loss_d_L[i+1] << std::endl;
            std::cout << "z^(" << i << "):         " << network.layers[i].z << std::endl;
        }*/
        
        for(auto k=network.layers.size()-1  ;k>=1;k--)
        {
            //std::cout << "\nk: " << k << std::endl;
            //std::cout << "dim z: " << network.layers[k].z.getndims()[0] << std::endl;
            Tensor<T,1> temp1=D_Activation<T,1>(network.layers[k].activation,network.layers[k].z);
           // std::cout << "dim temp1: " << temp1.getndims()[0] << std::endl;
            
            //std::cout << "dim d_Loss_d_L[k+1]: " << d_Loss_d_L[k].getndims()[0] << std::endl;
            Tensor<T,1> temp=elem_Mult<T,1>(d_Loss_d_L[k+1],temp1);
            //std::cout << "dim temp: " << temp.getndims()[0] << std::endl;

            //std::cout << "dim network.layers[k].weights: " << network.layers[k].weights.getndims()[0] << " " << network.layers[k].weights.getndims()[1] << std::endl;

           // std::cout << "Temp: " << temp << std::endl;
            //std::cout << "Weights: " << network.layers[k].weights << std::endl;
            Tensor<T,1> temp2=vec_Matrix_prod(temp,network.layers[k].weights);
            //std::cout << "Temp2: " << temp2 << std::endl;

            d_Loss_d_L[k]=vec_Matrix_prod(temp,network.layers[k].weights);//i know 0,0 looks strange, but according to my math its correct
           // std::cout << "d_Loss_d_L[" << k << "]: " << d_Loss_d_L[k] << std::endl;
        }
        for(auto i=1 ;i<dim.size();i++)
        {
            d_Loss_d_L[i].isnan();
            //std::cout << "d_Loss_d_L[" << i << "]: " << d_Loss_d_L[i] << std::endl;
        }

        //dL/dW^(k)_ij and dL/db^(k)_i
        
        for(auto k=0;k<network.layers.size()-1;k++)
        {
            //std::cout<<"Layer :"<<" "<<k<<std::endl;
            Tensor<T,1> temp1(network.layers[k].output.getndims());
            Tensor<T,1> temp2(network.layers[k].output.getndims());
            
            temp2 = D_Activation<T,1>(network.layers[k].activation,network.layers[k].z);
            
            temp1=elem_Mult<T,1>(d_Loss_d_L[k+1],temp2);
            
            sum_changes_of_weights_and_biases[k].del_b = sum_changes_of_weights_and_biases[k].del_b +temp1;

           // std::cout<<"Layer input"<<" "<<network.layers[k].input<<std::endl;
            for(auto i=0;i<sum_changes_of_weights_and_biases[k].del_W.getndims()[0];i++)
            {
                for(auto j=0;j<sum_changes_of_weights_and_biases[k].del_W.getndims()[1];j++)
                {
                    sum_changes_of_weights_and_biases[k].del_W(i,j)=
                    sum_changes_of_weights_and_biases[k].del_W(i,j)+
                    temp1[i]*network.layers[k].input[j];
                   // std::cout << "Changes of weights and biases: " << sum_changes_of_weights_and_biases[k].del_W(i,j) << std::endl;
                }
            }

            //std::cout<<"Changes of weight and bias:"<<" "<<sum_changes_of_weights_and_biases[k].del_W<<std::endl;
        }

       // std::cout << "Checkpoint 1" << std::endl;
        int M=network.layers.size()-1;
        Tensor<T,1> z_min_pred(network.layers[M].output.getndims());
        z_min_pred = network.layers[M].output-training_data.target;
        Tensor<T,1> L_prime(network.layers[M].output.getndims());
        L_prime = D_Loss<T,T>(network.output_size,Loss_function,network.layers[M].output,training_data.target);

        sum_changes_of_weights_and_biases[M].del_b = sum_changes_of_weights_and_biases[M].del_b +L_prime;
        sum_changes_of_weights_and_biases[M].del_b.isnan();
        sum_changes_of_weights_and_biases[M].del_W.isnan();
        L_prime.isnan();

       // std::cout << "Checkpoint 2" << std::endl;

       //  std::cout<<"Layer input"<<" "<<network.layers[M].input<<" "<<network.layers[M].output<<" "<<std::endl;
        
        for(auto i=0;i<sum_changes_of_weights_and_biases[M].del_W.getndims()[0];i++)
        {   
            for(auto j=0;j<sum_changes_of_weights_and_biases[M].del_W.getndims()[1];j++)
            {
                
                sum_changes_of_weights_and_biases[M].del_W(i,j)=
                sum_changes_of_weights_and_biases[M].del_W(i,j)+
                L_prime[i]*network.layers[M].input[j];
                
            }

        }

        //std::cout<<"Changes of weight and bias:"<<" "<<sum_changes_of_weights_and_biases[M].del_W<<std::endl;
        //std::cout << "Checkpoint 3" << std::endl;

        //sum_changes_of_weights_and_biases[M].del_W.isnan();
        for(auto i=0;i<network.layers.size();i++)
        {
            network.layers[i].weights.isnan("None Weight at End of Optimization detected");
            network.layers[i].bias.isnan("None Bias at End of Optimization detected");
        }
        //check if any sum_changes_of_weights_and_biases are nan
        for(auto i=0;i<sum_changes_of_weights_and_biases.size();i++)
        {
            sum_changes_of_weights_and_biases[i].del_W.isnan("None sum_changes Weights at End of Optimization detected");
            sum_changes_of_weights_and_biases[i].del_b.isnan("None sum_changes Biases atEnd of Optimization detected");
        }
        is_plausible_sum_changes_of_weights_and_biases("After optimization step");

    }

    void update_weights_and_biases_gradient_descent(T learning_rate=0.01)
    {
        //std::cout << "Updating weights and biases" << std::endl << std::endl << std::endl;
        for(auto k=0;k<network.layers.size();k++)
        {
            Tensor<T,1> temp_b(sum_changes_of_weights_and_biases[k].del_b.getndims());
            Tensor<T,2> temp_w(sum_changes_of_weights_and_biases[k].del_W.getndims());
            temp_b = sum_changes_of_weights_and_biases[k].del_b*learning_rate;
            temp_w = sum_changes_of_weights_and_biases[k].del_W*learning_rate;
            //std::cout << "temp_b: " << temp_b << std::endl;
            //std::cout << "temp_w: " << temp_w << std::endl;
            //std::cout << "network.layers[" << k << "].weights: " << network.layers[k].weights << std::endl;
            network.layers[k].weights = network.layers[k].weights - temp_w;
            network.layers[k].bias = network.layers[k].bias - temp_b;
           // std::cout << "network.layers[" << k << "].weights: " << network.layers[k].weights << std::endl;

        }
        //implement some gradients for the first aka last layer(output layer)
        
        for(auto k=0;k<network.layers.size();k++)
        {
            sum_changes_of_weights_and_biases[k].reset_Del_Weights_and_Bias();
        }
    }

    void update_weights_and_biases_adam(T learning_rate, T B1, T B2,T epsilon, int& iteration)
    {

        //check if sum_changes_of_weights_and_biases are nan
        for(auto i=0;i<sum_changes_of_weights_and_biases.size();i++)
        {
            sum_changes_of_weights_and_biases[i].del_W.isnan("None sum_changes Weights at Start of Adam detected");
            sum_changes_of_weights_and_biases[i].del_b.isnan("None sum_changes Biases at Start of Adam detected");
        }
        //check if weights or biases are nan
        for(auto i=0;i<network.layers.size();i++)
        {
            network.layers[i].weights.isnan("None Weight detected, going into Adam");
            network.layers[i].bias.isnan("None Bias detected, going into Adam");
        }
         iteration++;

        for(auto k=0;k<network.layers.size();k++)
        {
            Tensor<T,2> temp_g_w(sum_changes_of_weights_and_biases[k].del_W.getndims());
            Tensor<T,1> temp_g_b(sum_changes_of_weights_and_biases[k].del_b.getndims());

            //std::cout<<"Layer :"<<" "<<k<<std::endl;

             //std::cout<<"gradient:"<<" "<<sum_changes_of_weights_and_biases[k].del_W<<std::endl;

            temp_g_w=sum_changes_of_weights_and_biases[k].del_W;
            temp_g_b=sum_changes_of_weights_and_biases[k].del_b;

           

            //std::cout<<temp_g_w<<std::endl;

            temp_g_w.isnan("Nan detected at temp_g_w");
            temp_g_b.isnan("Nan detected at temp_g_b");

            // define elementwise square 

            Tensor tmp_w_square=square(temp_g_w);
            Tensor tmp_b_square=square(temp_g_b);

            tmp_w_square.isnan("Nan1");
            tmp_b_square.isnan("naa2");


            tmp_w_square*=(1.0-B2);
            tmp_b_square*=(1.0-B2);

            tmp_w_square.isnan("Nan1");
            tmp_b_square.isnan("naa2");

            //std::cout<<"tep:"<<" "<<temp_g_w<<std::endl;


            temp_g_w*=(1.0-B1);
            temp_g_b*=(1.0-B1);

            temp_g_w.isnan("Nan detected at temp_g_w");
            temp_g_b.isnan("Nan detected at temp_g_b");

           // std::cout<<"tep:"<<" "<<temp_g_w<<std::endl;



            Tensor<T,2> tmp1_add_w=B1*sum_changes_of_weights_and_biases[k].mw;
            Tensor<T,2> tmp2_add_w=B2*sum_changes_of_weights_and_biases[k].vw;
            Tensor<T,1> tmp1_add_b=B1*sum_changes_of_weights_and_biases[k].mb;
            Tensor<T,1> tmp2_add_b=B2*sum_changes_of_weights_and_biases[k].vb;



            sum_changes_of_weights_and_biases[k].mw=tmp1_add_w+temp_g_w;
            sum_changes_of_weights_and_biases[k].vw=tmp2_add_w+tmp_w_square;
            
            sum_changes_of_weights_and_biases[k].mb=tmp1_add_b+temp_g_b;
            sum_changes_of_weights_and_biases[k].vb=tmp2_add_b+tmp_b_square;

            sum_changes_of_weights_and_biases[k].mw.isnan("is weights nan");
            sum_changes_of_weights_and_biases[k].vw.isnan("velocity weights nan");

            Tensor<T,2> mdashw=sum_changes_of_weights_and_biases[k].mw/(1.0f-powf(B1,static_cast<float>(iteration)));
            Tensor<T,1> mdashb=sum_changes_of_weights_and_biases[k].mb/(1.0f-powf(B1,static_cast<float>(iteration)));
            Tensor<T,2> vdashw=sum_changes_of_weights_and_biases[k].vw/(1.0f-powf(B2,static_cast<float>(iteration)));
            Tensor<T,1>  vdashb=sum_changes_of_weights_and_biases[k].vb/(1.0f-powf(B2,static_cast<float>(iteration)));

            mdashw.isnan("Nan in mdashw");
            vdashw.isnan("Nan in vdashw");
           // std::cout<<"m"<<" "<<mdashw<<" "<<1.0-std::pow(B2,iteration)<<std::endl;
            //std::cout<<"v"<<" "<<vdashw<<" "<<1.0-std::pow(B1,iteration)<<std::endl;

           

           // update all weights 

            for(auto i=0;i<network.layers[k].weights.getsize();i++)
            {

                network.layers[k].weights[i]=network.layers[k].weights[i]-learning_rate/std::sqrt(vdashw[i]+epsilon)*mdashw[i];


            }

            
            for(auto i=0;i<network.layers[k].bias.getsize();i++)
            {

                network.layers[k].bias[i]=network.layers[k].bias[i]-learning_rate/std::sqrt(vdashb[i]+epsilon)*mdashb[i];


            }

        }


        //check if any sum_changes_of_weights_and_biases are nan
        for(auto i=0;i<sum_changes_of_weights_and_biases.size();i++)
        {
            sum_changes_of_weights_and_biases[i].del_W.isnan("None sum_changes Weights at End of Adam detected");
            sum_changes_of_weights_and_biases[i].del_b.isnan("None sum_changes Biases at End of Adam detected");
        }
        //check if weights or biases are nan
        for(auto i=0;i<network.layers.size();i++)
        {
            network.layers[i].weights.isnan("None Weight at End of Adam detected");
            network.layers[i].bias.isnan("None Bias at End of Adam detected");
        }

        for(auto k=0;k<network.layers.size();k++)
        {
            sum_changes_of_weights_and_biases[k].reset_Del_Weights_and_Bias();
        }
    }

    void independent_optimization_step(Training_data<T> training_data);

    std::vector<Tensor<T,1>> independent_get_del_L_del_L_k(Training_data<T> training_data);

    std::vector<Tensor<T,2>> independent_get_del_L_del_W_ij(Training_data<T> training_data);

    std::vector<Tensor<T,1>> independent_get_del_L_del_b_i(Training_data<T> training_data);

    Tensor<T,2> get_free_del_f_alpha_del_z_beta(int Layernum/*counting from zero!*/);//returns del f^(k)_alpha/del z^(k)_beta
    
       
};

template<class T>
void Optimizer<T>::independent_optimization_step(Training_data<T> training_data)
{
    network.free_forward(training_data.input);
    //std::vector<Tensor<T,1>> del_L_del_L_k=independent_get_del_L_del_L_k(training_data);

    std::vector<Tensor<T,2>> del_L_del_W_ij=independent_get_del_L_del_W_ij(training_data);
    std::vector<Tensor<T,1>> del_L_del_b_i=independent_get_del_L_del_b_i(training_data);

    for(auto k=0;k<network.layers.size();k++)
    {
        sum_changes_of_weights_and_biases[k].del_b = sum_changes_of_weights_and_biases[k].del_b +del_L_del_b_i[k];
        sum_changes_of_weights_and_biases[k].del_W = sum_changes_of_weights_and_biases[k].del_W +del_L_del_W_ij[k];
    }
}

template<class T>
std::vector<Tensor<T,1>> Optimizer<T>::independent_get_del_L_del_L_k(Training_data<T> training_data)
{
    std::vector<Tensor<T,1>> d_Loss_d_L;
    std::vector<int> dim;//dimension of second index of W^(k)_ij
    dim.push_back(0);//this is just a filler! we need Dim(dL/dL^(k+1)) = Dim(z^(k))
    for(auto i=1;i<network.layers.size();i++)
    {
        dim.push_back(network.layers[i].num_of_neurons);
    }
    dim.push_back(network.output_size);
    //std::cout<< "Checkpoint 1" << std::endl;
    for(auto i=0;i<dim.size();i++)//the first entry is just a filler for easier arythmetics
    {      
        Tensor<T,1> temp(std::vector<T>(dim[i],0));
        d_Loss_d_L.push_back( temp);
    }
    //std::cout<< "Checkpoint 2" << std::endl;
    
    {
    Tensor<T,2> df_dz = get_free_del_f_alpha_del_z_beta(network.layers.size()-1);//needs testing
    //std::cout << "df_dz: " << df_dz << std::endl;
    Tensor<T,1> dLoss=D_Loss<T,T>(network.output_size,Loss_function,network.layers[network.layers.size()-1].output,training_data.target);
    //std::cout << "dLoss: " << dLoss << std::endl;
    Tensor<T,1> temp=vec_Matrix_prod(dLoss,df_dz);
    //std::cout << "temp: " << temp << std::endl;
    d_Loss_d_L[network.layers.size()] = vec_Matrix_prod(temp,network.layers[network.layers.size()-1].weights);
    }
    //std::cout<< "Checkpoint 3" << std::endl;
    
    for(auto k=network.layers.size()-1  ;k>=1;k--)
    {
        Tensor<T,2> df_dz = get_free_del_f_alpha_del_z_beta(k);//needs testing
        Tensor<T,1> dLoss=d_Loss_d_L[k+1];
        Tensor<T,1> temp=vec_Matrix_prod(dLoss,df_dz);
        d_Loss_d_L[k]=vec_Matrix_prod(temp,network.layers[k].weights);
    }
    
    return d_Loss_d_L;
}

template<class T>
std::vector<Tensor<T,2>> Optimizer<T>::independent_get_del_L_del_W_ij(Training_data<T> training_data)
{
    std::vector<Tensor<T,1>> d_Loss_d_L=independent_get_del_L_del_L_k(training_data);
    std::vector<Tensor<T,2>> del_L_del_W_ij;

    std::vector<Tensor<T,2>> df_dz;
    for(auto k=0;k<network.layers.size();k++)
    {
        Tensor<T,2> temp= get_free_del_f_alpha_del_z_beta(k);
        df_dz.push_back(temp);
    }

    for(auto k=0;k<network.layers.size()-1;k++)
    {
        Tensor<T,1> temp=vec_Matrix_prod(d_Loss_d_L[k+1],df_dz[k]);
        Tensor<T,2> temp2=outer_product(temp,network.layers[k].input);
        del_L_del_W_ij.push_back(temp2);
    }

    {
        Tensor<T,1> temp0 = D_Loss<T,T>(network.output_size,Loss_function,network.layers[network.layers.size()-1].output,training_data.target);
        Tensor<T,1> temp = vec_Matrix_prod(temp0,df_dz[network.layers.size()-1]);
        Tensor<T,2> temp2 = outer_product(temp,network.layers[network.layers.size()-1].input);
        del_L_del_W_ij.push_back(temp2);
    }

    return del_L_del_W_ij;
}

template<class T>
std::vector<Tensor<T,1>> Optimizer<T>::independent_get_del_L_del_b_i(Training_data<T> training_data)
{
    std::vector<Tensor<T,1>> d_Loss_d_L=independent_get_del_L_del_L_k(training_data);
    std::vector<Tensor<T,1>> del_L_del_b_i;

    std::vector<Tensor<T,2>> df_dz;
    for(auto k=0;k<network.layers.size();k++)
    {
        Tensor<T,2> temp= get_free_del_f_alpha_del_z_beta(k);
        df_dz.push_back(temp);
    }

    for(auto k=0;k<network.layers.size()-1;k++)
    {
        Tensor<T,1> temp=vec_Matrix_prod(d_Loss_d_L[k+1],df_dz[k]);
        del_L_del_b_i.push_back(temp);
    }

    {
        Tensor<T,1> temp0 = D_Loss<T,T>(network.output_size,Loss_function,network.layers[network.layers.size()-1].output,training_data.target);
        Tensor<T,1> temp = vec_Matrix_prod(temp0,df_dz[network.layers.size()-1]);
        del_L_del_b_i.push_back(temp);
    }

    return del_L_del_b_i;
}

template<class T>
Tensor<T,2> Optimizer<T>::get_free_del_f_alpha_del_z_beta(int Layernum/*counting from zero!*/)//returns del f^(k)_alpha/del z^(k)_beta
{
    int dim_alpha = network.layers[Layernum].num_of_neurons;//gives dim of alpha
    int dim_z = network.layers[Layernum].z.getsize();//gives dim of beta
    Tensor<T,2> temp(std::vector<int>{dim_alpha,dim_z});
    double h=1e-2;
    for(auto i=0;i<dim_alpha;i++)
    {
        for(auto j=0;j<dim_z;j++)
        {
            Tensor<T,1> z_plus(network.layers[Layernum].z);
            Tensor<T,1> z_minus(network.layers[Layernum].z);
            z_plus[j]=z_plus[j]+h;
            z_minus[j]=z_minus[j]-h;
            double f_plus = network.layers[Layernum].free_funcs[i](z_plus);
            double f_minus = network.layers[Layernum].free_funcs[i](z_minus);
            //if(f_plus!=f_minus)
            //std::cout << "f_plus, f_minus: " << f_plus << " " << f_minus << std::endl;
            double temp1 = (f_plus-f_minus)/(2*h);

            temp(i,j)=T(temp1);
           // std::cout << "\nLayer: " << Layernum << std::endl;
           // std::cout << "z_plus: " << z_plus << std::endl;
           if(0)
           if(f_plus!=f_minus)
           {
                std::cout << "\nLayer: " << Layernum << std::endl;
                std::cout << "i,j: " << i << " " << j << std::endl;
                std::cout << "z_plus: " << z_plus << std::endl;
                std::cout << "f_plus: " << f_plus << std::endl;
                std::cout << "f_minus: " << f_minus << std::endl;
                std::cout << "temp1: " << temp1 << std::endl;
           }

        }
    }
    temp.isnan("get_free_del_f_alpha_del_z_beta contains nan values");
    temp.is_plausible("get_free_del_f_alpha_del_z_beta contains suspicious values");
    temp.isnan_or_inf("get_free_del_f_alpha_del_z_beta contains inf values");
    return temp;
}




#endif