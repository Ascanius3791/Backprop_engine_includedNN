// definition of various activation functions 
// Author : Jakob Hoffmann 
#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP
#include"Tensor.hpp"
#include<cmath>


template<class T,std::size_t N=2>
Tensor<T,N> Activation(std::string type,Tensor<T,N>& T1) 
{


    Tensor<T,N> res(T1.getndims());

     if(type=="ReLU")
     {
        res=RELU(T1);
     }

     else if(type=="tanh")
     {
        res=TANH(T1);
     }

     else if(type=="Sigmoid")
     {
       res=SIGMOID(T1);
     }

     else
     {
        throw std::runtime_error("Activation function not defined");
     }



    return res;


   


}

template<class T, std::size_t N=2>
Tensor<T,N> D_Activation(std::string type,Tensor<T,N>& T1) 
{
   Tensor<T,N> res(T1.getndims());

     if(type=="ReLU" || type=="D_ReLU")
     {
        res=D_RELU(T1);
     }

     else if(type=="tanh" || type=="D_tanh")
     {
        res=D_TANH(T1);
     }

     else if(type=="Sigmoid" || type=="D_Sigmoid")
     {
       res=D_SIGMOID(T1);
     }

     else
     {
        throw std::runtime_error("Activation function not defined");
     }

      return res;
}
     

template<class T, std::size_t N=2>
Tensor<T,N> RELU( Tensor<T,N>& Tens1) 
{

    Tensor<T,N> res(Tens1.getndims());

    for(auto i=0;i<Tens1.getsize();i++)
    {

         if(Tens1[i]>0.0)
         {
            res[i]=Tens1[i];
         }
         else
         {
            res[i]=0.0;
         }
    }

    //std::cout<<res<<std::endl;




    return res;
}

template<class T, std::size_t N=2>
Tensor<T,N> TANH( Tensor<T,N>& Tens1)
{

    Tensor<T,N> res(Tens1.getndims());
    for(auto i=0;i<Tens1.getsize();i++)
    {
         res[i]=std::tanh(Tens1[i]);
    }

    return res; 
}

template<class T, std::size_t N=2>
Tensor<T,N> SIGMOID( Tensor<T,N>& Tens1)
{

    Tensor<T,N> res(Tens1.getndims());
    for(auto i=0;i<Tens1.getsize();i++)
    {
         res[i]=1.0/(1.0+std::exp(-Tens1[i]));
    }

    return res; 
}

template<class T, std::size_t N=2>
Tensor<T,N> D_RELU( Tensor<T,N>& Tens1) 
{

    Tensor<T,N> res(Tens1.getndims());

    for(auto i=0;i<Tens1.getsize();i++)
    {

         if(Tens1[i]>0.0)
         {
            res[i]=1.0;
         }
         else
         {
            res[i]=0.0;
         }
    }

    //std::cout<<res<<std::endl;
    return res;
}

template<class T, std::size_t N=2>
Tensor<T,N> D_TANH( Tensor<T,N>& Tens1)
{

    Tensor<T,N> res(Tens1.getndims());
    for(auto i=0;i<Tens1.getsize();i++)
    {
         res[i]=std::pow(std::cosh(Tens1[i]),-2.0);
    }

    return res; 
}

template<class T, std::size_t N=2>
Tensor<T,N> D_SIGMOID( Tensor<T,N>& Tens1)
{

    Tensor<T,N> res(Tens1.getndims());
    for(auto i=0;i<Tens1.getsize();i++)
    {
         res[i]=std::exp(-Tens1[i])/(std::pow(1.0+std::exp(-Tens1[i]),2));
    }

    return res; 
}


template<class T>
Tensor<T,1> free_Activation(std::vector<T(*)(Tensor<T,1>)> f,Tensor<T,1>& T1) //takes array of functions and applies them to a tensor, returns a tensor<T,1>
{
      Tensor<T,1> res(T1.getndims());
      for(auto i=0;i<T1.getsize();i++)
      {
          res[i]=f[i](T1);
      }
    return res;
}

#endif