// test Activation function 
#define BOOST_TEST_MODULE example
#include "../lib/Activation.hpp"
#include <boost/test/included/unit_test.hpp>
#include<iostream>


BOOST_AUTO_TEST_CASE( test)
{

    // test RelU function 

    std::vector<double> v{7.8,0.8,7.8,5.6,-7.8};

    Tensor<double,1> tens(v);

    std::cout<<tens<<std::endl;


    Tensor relu=Activation<double,1>("ReLU",tens);

    std::cout<<relu<<std::endl;

    BOOST_CHECK_EQUAL(tens(0),relu(0));

    Tensor<double,2> tens2(std::vector<std::vector<double>>{{8.9,7.8,0.0},{6.7,-8.9,5.6}});

    Tensor relu2=Activation("ReLU",tens2);

    std::cout<<relu2<<std::endl;

    Tensor tanh1=Activation("tanh",tens);

    for(auto i=0;i<tanh1.getsize();i++)
    {
        BOOST_CHECK_EQUAL(tanh1(i),std::tanh(tens(i)));

    }

    std::cout<<tanh1;

    Tensor tanh2=Activation("tanh",tens2);

    for(auto i=0;i<tanh2.getsize();i++)
    {
        BOOST_CHECK_EQUAL(tanh2[i],std::tanh(tens2[i]));
    }

    std::cout<<tanh2;


















}
