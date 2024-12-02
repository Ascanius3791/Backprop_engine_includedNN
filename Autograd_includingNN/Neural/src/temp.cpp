#include <iostream>
#include <vector>
#include "../lib/Tensor.hpp"

int main()
{
    std::vector<int> dims{3};
    Tensor<double,1> Tens(dims);
    Tens(1)=4.8;
    Tens(2)=5.8;
    Tens(0)=3.8;

    Tensor<double,1> Tens2 = Tens;
    Tensor<double,1> Tens3 = Tens2+Tens;
    for(auto i=0;i<dims[0];i++)
    {
        std::cout << Tens(i) << std::endl;
    }
    std::cout << Tens3 << std::endl;
    return 0;
}