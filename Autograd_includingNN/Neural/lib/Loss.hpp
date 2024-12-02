// definition of various loss functions 
#ifndef LOSS_HPP
#define LOSS_HPP

#include<iostream>
#include<vector>
#include "Tensor.hpp"


template<class T, typename R>//R is the return type
R square( Tensor<T,1>& Prediction, Tensor<T,1>& Target) 
{

    R sum = 0.0;

    for(auto i=0;i<Prediction.getsize();i++)
    {
        sum+=std::pow((Prediction[i]-Target[i]),2);
    }

    sum= sum/2;

    return sum;
}

template<class T, typename R>//R is the return type
Tensor<R,1> D_square(std::size_t dim, Tensor<T,1>& Prediction, Tensor<T,1>& Target) //returns componentwise derivative of square loss aka nabla
{

    if(Prediction.getsize()!=Target.getsize() || Prediction.getsize()!=dim)
    {
        throw std::runtime_error("dimensions of tensors do not match");
    }
    
    Tensor<R,1> Nabla(std::vector<R>(dim,R(0)));
    

    for(auto i=0;i<Prediction.getsize();i++)
    {
        Nabla[i]=(Prediction[i]-Target[i]);
    }

    return Nabla;
}

template<class T, typename R>//R is the return type
R mean_abs( Tensor<T,1>& Prediction, Tensor<T,1>& Target) 
{

    R sum = 0.0;

    for(auto i=0;i<Prediction.getsize();i++)
    {
        sum+=abs(Prediction[i]-Target[i]);
    }

    sum= sum/Prediction.getsize();

    return sum;
}

template<class T, typename R>//R is the return type
Tensor<R,1> D_mean_abs(std::size_t dim, Tensor<T,1>& Prediction, Tensor<T,1>& Target) //returns componentwise derivative of mean_abs loss aka nabla
{

    if(Prediction.getsize()!=Target.getsize() || Prediction.getsize()!=dim)
    {
        throw std::runtime_error("dimensions of tensors do not match");
    }
    
    Tensor<R,1> Nabla(std::vector<R>(dim,R(0)));

    for(auto i=0;i<Prediction.getsize();i++)
    {
        R temp =(Prediction[i]-Target[i]);
        if(temp>0)
        Nabla[i]=1;
        else if(temp<0)
        Nabla[i]=-1;
        else
        Nabla[i]=0;
    }

    return Nabla*(1/dim);
}

template<class T, typename R>
R free_Loss_f(std::size_t dim, Tensor<T,1>& Prediction, Tensor<T,1>& Target)
{
    
    if(dim!=3)
    {
        throw std::runtime_error("Loss function only defined for dim=3");
    }
    return Prediction[0]*Prediction[1];
}

template<class T, typename R>
Tensor<R,1> free_D_Loss_f(std::size_t dim, Tensor<T,1>& Prediction, Tensor<T,1>& Target)
{
    if(dim!=3)
    {
        throw std::runtime_error("Loss function only defined for dim=3");
    }
    Tensor<R,1> Nabla(std::vector<R>(dim,R(0)));
    Nabla[0]=Prediction[1];
    Nabla[1]=Prediction[0];
    Nabla[2]=0;
    return Nabla;
}

template<class T, typename R>
R free_Loss_g(std::size_t dim, Tensor<T,1>& Prediction, Tensor<T,1>& Target)
{
    if(dim!=2)
    {
        throw std::runtime_error("Loss function only defined for dim=2");
    }
    return Prediction[0]*Prediction[1];
}

template<class T, typename R>
Tensor<R,1> free_D_Loss_g(std::size_t dim, Tensor<T,1>& Prediction, Tensor<T,1>& Target)
{
    if(dim!=2)
    {
        throw std::runtime_error("Loss function only defined for dim=2");
    }
    Tensor<R,1> Nabla(std::vector<R>(dim,R(0)));
    Nabla[0]=Prediction[1];
    Nabla[1]=Prediction[0];
    return Nabla;
}

template<class T, typename R>
R free_Loss_h(std::size_t dim, Tensor<T,1>& Prediction, Tensor<T,1>& Target)
{
    if(dim!=3)
    {
        throw std::runtime_error("Loss function only defined for dim=3");
    }
    return Prediction[0]+Prediction[1]+Prediction[2];
}

template<class T, typename R>
Tensor<R,1> free_D_Loss_h(std::size_t dim, Tensor<T,1>& Prediction, Tensor<T,1>& Target)
{
    if(dim!=3)
    {
        throw std::runtime_error("Loss function only defined for dim=3");
    }
    Tensor<R,1> Nabla(std::vector<R>(dim,R(0)));
    Nabla[0]=1;
    Nabla[1]=1;
    Nabla[2]=1;
    return Nabla;
}


template<class T, typename R>//R is the return type
R Loss(std::string type,Tensor<T,1>& Prediction, Tensor<T,1>& Target) 
{
    R sum = 0.0;

    if(Prediction.getsize()!=Target.getsize())
    {
        //std::cout << "Prediction size: " << Prediction.getsize() << std::endl;
        throw std::runtime_error("dimensions of tensors do not match");
    }

    if(type=="square")
    {
        sum=square(Prediction,Target);
    }

    else if(type=="mean_abs")
    {
        sum=mean_abs(Prediction,Target);
    }

    else if(type=="free_Loss_f")
    {
        sum=free_Loss_f(Prediction,Target);
    }

    else if(type=="free_Loss_g")
    {
        sum=free_Loss_g(Prediction,Target);
    }

    else if(type=="free_Loss_h")
    {
        sum=free_Loss_h(Prediction,Target);
    }

    else
    {
        throw std::runtime_error("Loss function not defined");
    }

    return sum;
}   

template<class T, typename R>//R is the return type
Tensor<R,1> D_Loss(std::size_t dim, std::string type,Tensor<T,1>& Prediction, Tensor<T,1>& Target) 
{
    Tensor<R,1> Nabla(std::vector<R>(dim,R(0)));

    if(Prediction.getsize()!=Target.getsize() || Prediction.getsize()!=dim)
    {
        if(Prediction.getsize()!=Target.getsize())
        {
            throw std::runtime_error("Prediction size: " + std::to_string(Prediction.getsize()) + " Target size: " + std::to_string(Target.getsize()));
        }
        if(Prediction.getsize()!=dim)
        {
            throw std::runtime_error("Prediction size: " + std::to_string(Prediction.getsize()) + " dim: " + std::to_string(dim));
        }
        throw std::runtime_error("dimensions of tensors do not match");
    }

    if(type=="square" || type=="D_square")
    {
        Nabla=D_square<T,R>(dim,Prediction,Target);
    }

    else if(type=="mean_abs" || type=="D_mean_abs")
    {
        Nabla=D_mean_abs<T,R>(dim,Prediction,Target);
    }

    else if(type=="free_Loss_f" || type=="free_D_Loss_f")
    {
        Nabla=free_D_Loss_f<T,R>(dim,Prediction,Target);
    }

    else if(type=="free_Loss_g" || type=="free_D_Loss_g")
    {
        Nabla=free_D_Loss_g<T,R>(dim,Prediction,Target);
    }

    else if(type=="free_Loss_h" || type=="free_D_Loss_h")
    {
        Nabla=free_D_Loss_h<T,R>(dim,Prediction,Target);
    }

    else
    {
        throw std::runtime_error("Loss function not defined");
    }
    //std::cout << "Nabla_direkt: " << Nabla << std::endl;
    return Nabla;
}


#endif