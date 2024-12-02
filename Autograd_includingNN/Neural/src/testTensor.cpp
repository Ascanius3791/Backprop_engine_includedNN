// this is a c++ program to test the class Tensor 
#define BOOST_TEST_MODULE example
#include <boost/test/included/unit_test.hpp>
#include<iostream>
#include "../lib/Tensor.hpp"


BOOST_AUTO_TEST_CASE( test)
{
    // Tensor(....)
    int i=1;
    int j=1;
    BOOST_REQUIRE_EQUAL(i,j);

    std::vector<int> v{1,2,3};
    std::vector<int> v2{1,2,3};

    std::cout<<"test vector comparison"<<std::endl;



    BOOST_CHECK_EQUAL_COLLECTIONS(&v[0],&v[0]+3,&v2[0],&v2[0]+3);

    std::vector<int> dims{3,3,3};

    Tensor<double,3> Tens(dims);

    double g=Tens(1,1,2);

    Tens(2,1,2)=4.8;

    std::cout<<Tens(2,1,2)<<std::endl;
    std::cout<<Tens<<std::endl;

    Tensor<double,0> Tens2(6.0);

    std::cout<<"Scalar:"<<Tens2<<std::endl;

    std::vector<double> vec{9.8,7.8,5.8};

    Tensor<double,1> Tens3(vec);

    std::cout<<"Vector:"<<Tens3<<std::endl;

    double gnew=Tens3(1);

    for(auto i=0;i<vec.size();i++)
    {
        BOOST_CHECK_EQUAL(vec[i],Tens3(i));
    }

    //Tens3(0)=9.3;

    double *array=Tens3.to_array();

    BOOST_CHECK_EQUAL_COLLECTIONS(array,array+3,&vec[0],&vec[0]+3);

    std::cout<<"Vector:"<<Tens3(0)<<" "<<Tens3(1)<<" "<<Tens3(2)<<std::endl;

    std::vector<std::vector<double>> vec2{{9.7,6.7},{5.6,7.8}};

    Tensor<double,2> tens(vec2);

    std::cout<<tens(1,1)<<std::endl;

    Tensor<double,2> C2(tens.getndims());
    C2=tens;
    std::cout<<C2<<std::endl;

    Tensor Tens5=tens*5.0;

    BOOST_CHECK_EQUAL(Tens5(0,0),5.0*9.7);

    std::cout<<Tens5(1,0)<<" "<<5.6*5.0<<std::endl;

    std::cout<<Tens5<<std::endl;


   // test scalar product    

    double T6=scalar_product(Tens3,Tens3);

    BOOST_CHECK_EQUAL(T6,9.8*9.8+7.8*7.8+5.8*5.8);

     // test contraction 


    Tensor T7=contract<double,2,1,1>(tens,Tens3,1,0);

    std::cout<<T7<<std::endl;

    BOOST_CHECK_EQUAL(T7(0),tens(0,0)*Tens3(0)+tens(0,1)*Tens3(1));

    Tensor T8=contract<double,2,2,2>(tens,tens,1,0);

    BOOST_CHECK_EQUAL(T8(0,0),tens(0,0)*tens(0,0)+tens(0,1)*tens(1,0));
    BOOST_CHECK_EQUAL(T8(0,1),tens(0,0)*tens(0,1)+tens(0,1)*tens(1,1));

    std::cout<<T8<<std::endl;
    Tensor T9=T8.square();

    std::cout<<T9<<" "<<std::pow(131.61,2.0)<<std::endl;

    //T9*=8.0;

    Tensor T10=T9.sqrt();

    std::cout<<T10<<std::endl;

    Tensor T12=T10/(6.0-4.0);

    std::cout<<T12<<std::endl;

}