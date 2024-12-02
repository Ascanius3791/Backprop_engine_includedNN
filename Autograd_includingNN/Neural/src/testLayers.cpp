#include"../lib/Layers.hpp"

int main()
{
    //contract matrix [[1 0 0 0 ],[0 1 0 0 ]] with vector [0.5 0.5 0 0 ] using tensors
    Tensor<float,2> matrix(std::vector<std::vector<float>>{{1,0,0,0},{0,1,0,0}});
    Tensor<float,1> vec(std::vector<float>{0.5,0.5,0,0});
    
    std::cout << "Vector: " << vec << std::endl;
    std::cout << "Matrix: " << matrix << std::endl;
    Tensor<float,1> result=contract<float,2,1,1>(matrix,vec,1,0);
    std::cout << "result: " << result << std::endl;
}