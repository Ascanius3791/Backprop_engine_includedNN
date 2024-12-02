// This is a c++ tensor class 
// Author: Jakob Hoffmann
// last edited: 11.11.2024 14:40 

// we store all input data and output data in tensor objects 

#ifndef TENSOR_HPP 
#define TENSOR_HPP

#include<vector>
#include<stdexcept>
#include<cassert>
#include<cmath>
#include<iostream>



// default tensot is a (0,2) tensor
template<class T, std::size_t N=2>
class Tensor{


    protected:
    std::vector<int> Ndims; // number of dimensions for every index (depends on number of input)
    T* data;               // store raw data in a pointer 
    int size;


    // get superindex from variadic template arguments 

    template<class...Indices>
    const int superindex(const Indices&... index)
    {


        if(sizeof...(Indices)!=this->ndimsize())
        {
            throw std::runtime_error("indices do not match");
        }
        
         // check if number of indices match with tensor dimensions 

        std::vector<int> ind;
        (ind.push_back(index),...); // fill vector with variadic template arguments;

        int linear=0;
        int factor=1;
        for(auto i=0;i<ind.size();i++)
        {
            if(ind[i]>=Ndims[i])
            {
                throw std::runtime_error("element not accesible");
            }
            linear+=ind[i]*factor;

            factor*=Ndims[i];
            
        }

        return linear;
    }

    
    const int superindex(const std::vector<int>& index)
    {
        int linear=0;
        int factor=1;
        for(auto i=0;i<index.size();i++)
        {
            if(index[i]>=Ndims[i])
            {
                for(auto i=0;i<index.size();i++)
                {
                    std::cout<<"index[i= " << i << "] : " << index[i]<<std::endl;
                    std::cout<<"Ndims[i= " << i << "] : " << Ndims[i]<<std::endl<<std::endl;
                }
                throw std::runtime_error("element not accesible");
            }

            linear+=index[i]*factor;
            factor*=Ndims[i];

        }

        return linear;
    }

    // get total dimension of tensor 

    public:

    constexpr int ndimsize()
    {
        return Ndims.size();
    }

    int getsize() const
    {
        return size;
    }



    // print tensor 

    



    //get reverse superindex 
    


    // default constructor 

    Tensor()=default;

    // constuctor taking tuple of indices 

    Tensor(std::vector<int> Ndims_);


    // constructor for dim 1 Tensor to fill from std::vector 
    Tensor(const std::vector<T> vec);

    //constructor for dim 2 Tensor to fill from std::vector<std::vector<T>>> 

    Tensor(const std::vector<std::vector<T>> vec2);

    // constructor for dim 0 Tensor 

    Tensor(const T scalar);


    // copy constructor 

    Tensor(const Tensor& tens)
    {
        this->size=tens.size;
        this->Ndims=tens.Ndims;
        data=new T[size];
        for(auto i=0;i<size;i++)
        {
            data[i]=tens.data[i];
        }
    }

    // move constructor 

    Tensor(Tensor&& tens)
    {
       
        this->Ndims=tens.Ndims;
        this->size=tens.size;

        std::swap(data,tens.data);
    }

    // assignment operator to set two tensors equal 

    Tensor& operator=(const Tensor& T2)
    {

        // self assignment 

        if(this==&T2)
        {
            return *this;
        }

        for(auto i=0;i<T2.getsize();i++)
        {
            if(std::isnan(T2.data[i]))
            {
                for(auto &j : T2.getndims())
                {
                    std::cout<<j<<" ";
                }
                throw std::runtime_error("nan detected in assignment");
            }
        }
        if(size!=T2.size)
        {
            std::cout<<"size1:"<<size<<std::endl;
            std::cout<<"size2:"<<T2.size<<std::endl;
            throw std::runtime_error("dimensions for assignment do not match");
        }

       this->size=T2.size;
       this->Ndims=T2.Ndims;
       for(auto i=0;i<T2.size;i++)
       {
         this->data[i]=T2.data[i];
       }
       
       
        return *this;
    }

    /*Tensor& operator=(Tensor&& T2)
    {
        if(this != &T2)
        {
            delete[] data;

            data=T2.data;
            size=T2.size;
        }

        return *this;
    }*/

    // destructor 

    ~Tensor()
    {
        delete[] data;
    }

    // constructor to fill from raw array 


    

    // opeartor to get index 

    template<class...Indices>
    T& operator()(const Indices...index)
    {
        return data[superindex(index...)];
    }

    T& operator()(const std::vector<int> Indices)
    {
        return data[superindex(Indices)];
    }

    T& operator[](const int i)
    {
        return data[i];
    }




   // resize tensor to new dimensions 

    void resize(std::vector<int> newdims);

    void resize(const int Ndim=2,int Nd=2); 

    // TODO: implement functions to easily handle data 

    friend std::ostream& operator<<(std::ostream& os, Tensor<T,N>& A)
    {
        // get index to print with 

        std::vector<int> index(A.getndims().size());

        // print vector or scalar 

        if(A.getndims().size()==1)
        {
           if(A.getndims()[0]==1)
           {
             os<<A(0)<<std::endl;
           }

           else
           {
            os<<"[";
            for(auto i=0;i<A.getndims()[0];i++)
            {
                os<<A(i)<<" ";
            }

            os<<"]";
           }
        }



        // print matrix

        else if(A.getndims().size()==2)
        {
            os<<"[";
            for(auto i=0;i<A.getndims()[0];i++)
            {
                os<<"[";

                for(auto j=0;j<A.getndims()[1];j++)
                {
                    os<<A(i,j)<<" ";
                }

                if(i!=A.getndims()[0]-1)
                {
                    os<<"],";
                }
                else
                {
                    os<<"]";
                }
            }

            os<<"]";
        }

        else if(A.getndims().size()==3)
        {


            
            os<<"[";
            for(auto i=0;i<A.getndims()[0];i++)
            {
                os<<"[";

                for(auto j=0;j<A.getndims()[1];j++)
                {

                    os<<"[";

                    for(auto k=0;k<A.getndims()[2];k++)
                    {

                         os<<A(i,j,k)<<" ";
                    }

                    if(j!=A.getndims()[1]-1)
                    {
                        os<<"],";
                    }
                    else
                    {
                        os<<"]";
                    }
                   
                }

                if(i!=A.getndims()[0]-1)
                {
                    os<<"],";
                }
                else
                {
                    os<<"]";
                }
            }

            os<<"]";

        }

        else
        {
            throw std::runtime_error("tensor not printable");
        }

        return os;

    }
    


    // convert tensor o raw_array 

    T* to_array();

    // copy tensor to device

    // getters 

    std::vector<int> getndims() const 
    {
        return Ndims;

    }

    // contract tensors over multiple indices 

    Tensor& contract(const std::vector<int> index);

    // contract over one specific index 
    Tensor& contract(const unsigned int index);

    // apply generic function to all tensor components 

    template<class F> 
    Tensor& apply(F function);


    // get all class operations 
    Tensor& operator+=(const T& val)
    {
        for(auto i=0;i<size;i++)
        {
            this->data[i]=this->data[i]+val;
        }

        return *this;
    }

    Tensor& operator-=(const T& val)
    {
         for(auto i=0;i<size;i++)
        {
            this->data[i]=this->data[i]-val;
        }

        return *this;

    }

    Tensor& operator*=(const T& val)
    {
        for(auto i=0;i<size;i++)
        {
            data[i]=val*data[i];
        }

        return *this;
    }

    Tensor& operator/=(const T& val)
    {
        for(auto i=0;i<size;i++)
        {
            data[i]=data[i]/val;
        }
        return *this;
    }

    // element wise square 

     Tensor square() const
    {
        Tensor result(Ndims);

        for(auto i=0;i<size;i++)
        {
            result[i]=data[i]*data[i];
        }

        return result;
    }

    // eement wise sqrt 

    Tensor sqrt() const
    {

         Tensor result(Ndims);

        for(auto i=0;i<size;i++)
        {
            result[i]=std::sqrt(data[i]);
        }

        return result;




    }

    void isnan(std::string message="")
    {
        for(auto i=0;i<size;i++)
        {
            if(std::isnan(data[i]))
            {
                if(message!="")
                {
                    std::cout<<message<<std::endl;
                }
                std::cout<<"Tensor with nans:"<<" "<<this[0]<<std::endl;
                throw std::runtime_error("Tensor has nans");
            }
        }
    }

    void isnan_or_inf(std::string message="")
    {

        for(auto i=0;i<size;i++)
        {
            if(std::isnan(data[i]) || std::isinf(data[i]))
            {
                std::cout<<"data[i]:"<<data[i]<<std::endl;
                if(message!="")
                {
                    std::cout<<message<<std::endl;
                }
                if(std::isnan(data[i]))
                {
                    std::cout<<"Tensor with nans:"<<" "<<this[0]<<std::endl;
                    throw std::runtime_error("Tensor has nans");
                }
                else
                {
                    std::cout<<"Tensor with infs:"<<" "<<this[0]<<std::endl;
                    throw std::runtime_error("Tensor has infs");
                }
            }
        }
    }

    void is_plausible(std::string message="")
    {
        T threshhold=1e3;
        for(auto i=0;i<size;i++)
        {
            if(std::abs(data[i])>threshhold)
            {

                std::cout<<"data[i]:"<<data[i]<<std::endl;
                if(message!="")
                {
                    std::cout<<message<<std::endl;
                }
                throw std::runtime_error("Plausibility check failed");
            }
        }
    }






};

// non member functions of tensor class 


// implementation of constructors 

template<class T, std::size_t N>
Tensor<T,N>::Tensor(std::vector<int> Ndims_)
{
    // first reserve space 
    // get number of dimensions first 

    if(Ndims_.size()!=N)
    {
        std::cout << "N ="  << N <<  std::endl << std::endl;
        for(auto i=0;i<Ndims_.size();i++)
        std::cout << Ndims_[i] << std::endl;
        throw std::runtime_error("sizes are not compatble");
    }

    int Ncount=1;
    for(const auto& elem : Ndims_)
    {
        if(elem<=0)
        {
            std::runtime_error("Negative dimension not allowed"); 
        }
        Ncount*=elem;
    }

    data=new T[Ncount];
    Ndims=Ndims_;
    size=Ncount;
}

// construction from a scalar 

template<class T,std::size_t N>
Tensor<T,N>::Tensor(const T scalar)
{

    static_assert(N==0,"dimension not allowed");
    // 0 dimensional tensor (only scalar number)
    data=new T[1];
    size=1;
    data[0]=scalar;
    Ndims.push_back(1);
}


// construction of (0,1) tensor from vector 
template<class T,std::size_t N>
Tensor<T,N>::Tensor(const std::vector<T> vec)
{

   static_assert(N==1,"dimension not allowed");
   //std::cout<<"size:"<<" "<<vec.size()<<std::endl;


   for(auto & elem : vec)
   {
     if(std::isnan(elem))
     {
        for(auto& j : vec)
        {
            std::cout<<j<<" ";
        }
        throw std::runtime_error("nan detected in vector");
     }
   }

   Ndims.push_back(vec.size());

   data=new T[vec.size()];

   size=vec.size();

   for(auto i=0;i<vec.size();i++)
   {
     data[i]=vec[i];
   }


}

// constructor for (0,2) tensor which takes a 2d vector as an agrument 

template<class T,std::size_t N>
Tensor<T,N>::Tensor(const std::vector<std::vector<T>> vec2)
{
    // check that rank is 2 
    static_assert(N==2,"dimension not allowed");
    // define number of dimensions 
    Ndims.push_back(vec2.size());
    Ndims.push_back(vec2[0].size());

    // fill data with values of vec2
    data=new T[Ndims[0]*Ndims[1]];

    size=Ndims[0]*Ndims[1];

    for(auto i=0;i<vec2.size();i++)
    {
        for(auto j=0;j<vec2[0].size();j++)
        {
            data[superindex(i,j)]=vec2[i][j];
        }
    }

}



template<class T,std::size_t N>
T* Tensor<T,N>::to_array()
{
    return data;

}

// implementation of resize function 

/*template<class T,std::size_t N>
void Tensor<T,N>::resize(std::vector<int> newdims)
{
    int Ncount=1;
    for(const auto& elem : newdims)
    {
        if(elem<=0)
        {
            // throw exception if any dimension is negative 
        }
        Ncount*=elem;
    }
    data.resize(Ncount);
    Ndims=newdims;

}*/


//apply function to all tensors 


template<class T, std::size_t N> template<class F>
Tensor<T,N>& Tensor<T,N>::apply(F function)
{

    for(auto i=0;i<size;i++)
    {
        data[i]=function(data[i]);
    }

    return *this;

}


// here are non member functions like addition 

template<class T,std::size_t N=2>
Tensor<T,N> operator+(Tensor<T,N>& Tens1,  Tensor<T,N>& Tens2)
{

    // check that tensors have exact same shape 
    for(auto i=0;i<Tens1.getndims().size();i++)
    {
        if(Tens1.getndims()[i]!=Tens2.getndims()[i])
        {
            throw std::runtime_error("dimensions of two tensors do not match");
        }
    }

    Tensor<T,N> result(Tens1.getndims());
    

    for(auto i=0;i<Tens1.getsize();i++)
    {
        result[i]=Tens1[i]+Tens2[i];
    }

    return result;
}

template<class T,std::size_t N=2>
Tensor<T,N> operator-(Tensor<T,N>& Tens1,  Tensor<T,N>& Tens2)
{

    // check that tensors have exact same shape 
    for(auto i=0;i<Tens1.getndims().size();i++)
    {
        if(Tens1.getndims()[i]!=Tens2.getndims()[i])
        {
            throw std::runtime_error("dimensions of two tensors do not match");
        }
    }

    Tensor<T,N> result(Tens1.getndims());
    

    for(auto i=0;i<Tens1.getsize();i++)
    {
        result[i]=Tens1[i]-Tens2[i];
    }

    return result;
}

// multiply with scalar number
template<class U, class T,std::size_t N=2>
Tensor<T,N> operator*(const U& scalar,Tensor<T,N>& Tens1)
{

    Tensor<T,N> Tens2(Tens1.getndims());
    for(auto i=0;i<Tens1.getsize();i++)
    {
        Tens2[i]=scalar*Tens1[i];
    }

    return Tens2;
}

template<class T, class U,std::size_t N=2>
Tensor<T,N> operator*(Tensor<T,N>& Tens1,const U& scalar)
{
    Tensor<T,N> Tens2(Tens1.getndims());
    for(auto i=0;i<Tens1.getsize();i++)
    {
        Tens2[i]=Tens1[i]*scalar;
    }

    return Tens2;
}

template<class T, class U,std::size_t N=2>
Tensor<T,N> operator/(Tensor<T,N>& Tens1,const U& scalar)
{
    Tensor<T,N> Tens2(Tens1.getndims());
    for(auto i=0;i<Tens1.getsize();i++)
    {
        Tens2[i]=Tens1[i]/scalar;
    }

    return Tens2;
}

template<class T, std::size_t N=2>
Tensor<T,N> elem_Divis(Tensor<T,N>& Tens0, Tensor<T,N>& Tens1)
{
    
    for(auto i=0;i<Tens0.getndims().size();i++)
    {
        if(Tens0.getndims()[i]!=Tens1.getndims()[i])
        {
            throw std::runtime_error("dimensions of two tensors do not match");
        }
    }

     Tensor<T,N> Tens2(Tens1.getndims());
    for(auto i=0;i<Tens1.getsize();i++)
    {
        Tens2[i]=Tens0[i]/Tens1[i];
    }

    return Tens2;



}

template<class T,std::size_t N=2>
Tensor<T,N> elem_Mult(Tensor<T,N>& Tens0,Tensor<T,N>& Tens1)
{
    for(auto i=0;i<Tens0.getndims().size();i++)
    {
        if(Tens0.getndims()[i]!=Tens1.getndims()[i])
        {
            throw std::runtime_error("dimensions of two tensors do not match");
        }
    }
    Tensor<T,N> Tens2(Tens1.getndims());
    for(auto i=0;i<Tens1.getsize();i++)
    {
        Tens2[i]=Tens0[i]*Tens1[i];
    }

    return Tens2;
}


//contract over arbitrary index of two different tensor , i..e A_ik=M_ij*M_jk

template<class T,std::size_t N=2, std::size_t M=2,std::size_t Nnew>
Tensor<T,Nnew> contract(Tensor<T,N>& Tens1, Tensor<T,M>& Tens2, const int index1, const int index2)
{
    // define some easy special cases 
    // check if dimension is >=1
    if(N<1 || M<1)
    {
        throw std::runtime_error("contraction not possible for given dimension");
    }

    // fix size of tensor 

    std::vector<int> newdims;

    // get non contracted indices of first tensor 

    for(auto i=0;i<Tens1.getndims().size();i++)
    {
        if(i!=index1)
        {
            newdims.push_back(Tens1.getndims()[i]);
        }

    }

    // get non contracted indices of second tensor 
    
    for(auto i=0;i<Tens2.getndims().size();i++)
    {
        if(i!=index2)
        {
            newdims.push_back(Tens2.getndims()[i]);
        }
    }
    
    //Tensor<T,Nnew> Tens;
    // left multiplication of scalar 

    Tensor<T,Nnew> Tens(newdims);


     // left matrix vector multiplication 


    if(N==1 & M==2)
    {
        int nnonctr=abs(static_cast<int>(Tens2.getndims().size()-index2));

        if(index1!=0 || index2>1)
        {
            throw std::runtime_error("contractions not possible");
        }

        for(auto i=0;i<Tens2.getndims()[nnonctr];i++)
        {
            T scalar=0.0;
            for(auto j=0;j<Tens2.getndims()[index2];j++)
            {
                std::vector<int> index{i,i};
                // pick out summation index 
                index[index2]=j;
                
                
                scalar+=Tens1(j)*Tens2(index);
            }

            Tens(i)=scalar;
        }
 
    }


     //  right matrix vector maultiplicaion 

    else if(N==2 & M==1)
    {

        // non contracted index 

        int nnonctr=abs(static_cast<int>(Tens1.getndims().size()-index1));

        if(index2!=0 || index1>1)
        {
            throw std::runtime_error("contractions not possible");
        }

        for(auto i=0;i<Tens1.getndims()[nnonctr];i++)
        {
            T scalar=0.0;
            for(auto j=0;j<Tens1.getndims()[index1];j++)
            {
                std::vector<int> index{i,i};
                // pick out summation index 
                index[index1]=j;
                
                std::cout << "We get here" << std::endl;
                //print superindex(it's private)
                //std::cout << Tens1.superindex(index) << std::endl;

                scalar+=Tens1(index)*Tens2(j);
                std::cout << "We dont get here" << std::endl;
            }
            Tens(i)=scalar;
        }

         
    }

//    // matrix matrix multiplication 

   else if(N==2 & M==2)
    {

        if(index2>1 || index1>1)
        {
            throw std::runtime_error("contractions not possible");
        }

        const int nnonctr1=abs(static_cast<int>(1-index1));
        const int nnonctr2=abs(static_cast<int>(1-index2));

        for(auto i=0;i<Tens1.getndims()[nnonctr1];i++)
        {
            for(auto j=0;j<Tens2.getndims()[nnonctr2];j++)
            {


                for(auto k=0;k<Tens1.getndims()[index1];k++)
                {
                    for(auto l=0;l<Tens2.getndims()[index2];l++)
                    {

                        
                        std::vector<int> ind1{i,i};
                        ind1[index1]=k;
                        std::vector<int> ind2{j,j};
                        ind2[index2]=l;


                        if(k==l)
                        {
                          Tens(i,j)+=Tens1(ind1)*Tens2(ind2);
                        }
                    }
                }

               
            }
        }

         
    }

    //currentl unavailable, dont use dim=1 for now
    /*
    else if(N==0)
    {
        // scalar with vector contraction arbitrary tensor
        Tens=Tens1(0)*Tens2;

    }
    
    else if(M==0)
    {
        // scalar with vector contraction arbitrary tensor
        Tens=Tens1*Tens2(0);

    }
    */



   else
   {
      throw std::runtime_error("Contraction not defined");
   }


    
   




    return Tens;

    //(N=0, M=1 or M=1 N=0) (scalar with vector contraction)

}

template<class T> 
Tensor<T,1> Matrix_vec_prod(Tensor<T,2>& Mat, Tensor<T,1>& Vec)
{
    if(Mat.getndims()[1]!=Vec.getndims()[0])
    {
        throw std::runtime_error("dimensions do not match");
    }
    Tensor<T,1> result(std::vector<T>(Mat.getndims()[0],0));
    for(auto i=0;i<Mat.getndims()[0];i++)
    {
        T scalar=0.0;
        for(auto j=0;j<Mat.getndims()[1];j++)
        {
            scalar+=Mat(i,j)*Vec(j);
        }

        result(i)=scalar;
    }

    return result;
}

template<class T>
Tensor<T,1> vec_Matrix_prod(Tensor<T,1>& Vec, Tensor<T,2>& Mat)
{   
    if(Mat.getndims()[0]!=Vec.getndims()[0])
    {
        throw std::runtime_error("dimensions do not match");
    }
    Tensor<T,1> result(std::vector<T>(Mat.getndims()[1],0));
    for(auto i=0;i<Mat.getndims()[1];i++)
    {
        T scalar=0.0;
        for(auto j=0;j<Mat.getndims()[0];j++)
        {
            scalar+=Vec(j)*Mat(j,i);
        }

        result(i)=scalar;
    }

    return result;
}
// scalar_product 

template<class T>
T scalar_product(Tensor<T,1>& Vec1, Tensor<T,1>& Vec2)
{
    T scalar=0.0;
    for(auto i=0;i<Vec1.getsize();i++)
    {
        scalar+=Vec1(i)*Vec2(i);
    }

    return scalar;
}

template<class T>
Tensor<T,2> outer_product(Tensor<T,1>& Vec1, Tensor<T,1>& Vec2)
{
    Tensor<T,2> result(std::vector<int>{Vec1.getsize(),Vec2.getsize()});
    for(auto i=0;i<Vec1.getsize();i++)
    {
        for(auto j=0;j<Vec2.getsize();j++)
        {
            result(i,j)=Vec1(i)*Vec2(j);
        }
    }

    return result;
}
// elementwise square of tensor

template<class T,std::size_t N=2>
Tensor<T,N> square(Tensor<T,N>& A)
{
    Tensor<T,N> result(A.getndims());
    for(auto i=0;i<A.getsize();i++)
    {
        result[i]=A[i]*A[i];
    }
    return result;
}








// const Tensor& Tens1 contract(const Tensor& Tens1, const Tensor& Tens2,int i, int j)
// {

// }

// contract(int i, int j)




// this is a c++ matrix class inherited from the Tensor class 
// template<class T>
// class Matrix : Tensor<T,2>
// {

//     // implement additional functions like inverse ...

// };

#endif 