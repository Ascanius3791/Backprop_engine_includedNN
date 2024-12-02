#include "../lib/Autograd.hpp"
#include<tuple>

const char func_name = 'h';

const int dim_for_Autograd_f =3, dim_for_Autograd_g=2, dim_for_Autograd_h=3;

double f(double x[2])
{
    double x_1=x[0];
    double x_2=x[1];
    
    //return log10(x[0]*x[1])*sin(x[1]);
    return log10(x_1*x_2)*sin(x_2);
}

double g(double x[2])
{
    double x_1=x[0];
    double x_2=x[1];
    
    return x_1*x_2*(x_1+x_2);
}

double h(double x[2])
{
    double x_1=x[0];
    return 3*x_1*x_1+4*x_1+2;
}

std::tuple<double*,int> nabla_f(double (*f)(double*),double* x,int N=2)
{
    double* nabla=new double[N];
    double h=1e-3;
    for(auto i=0;i<N;i++)
    {
        double* x_plus=new double[N];
        double* x_minus=new double[N];
        for(auto j=0;j<N;j++)
        {
            x_plus[j]=x[j];
            x_minus[j]=x[j];
        }
        x_plus[i]=x[i]+h;
        x_minus[i]=x[i]-h;
        nabla[i]=(f(x_plus)-f(x_minus))/(2*h);
    }
    return std::make_tuple(nabla,N);
}

template<class T>
T null_func(Tensor<T,1> T1)
{
    return 0.0;
}

template <class T, int Dim>
struct funcs_for_layer {
    int dim = Dim;
    std::vector<T (*)(Tensor<T, 1>)> A; // vector of function pointers

    // Constructor
    funcs_for_layer() {
        for (int i = 0; i < dim; ++i) {
            A.push_back(null_func);
        }
    }

};



template<class T>
struct funcs
{
    static T null_func(Tensor<T,1> T1)
    {
        return 0.0;
    }
    static T first_elem(Tensor<T,1> T1)
    {
        return T1[0];
    }
    static T second_elem(Tensor<T,1> T1)
    {
        return T1[1];
    }
    static T prod_first_second(Tensor<T,1> T1)
    {
        return T1[0]*T1[1];
    }
    static T log10_of_third(Tensor<T,1> T1)
    {
        return log10(T1[2]);
    }
    static T sin_of_second(Tensor<T,1> T1)
    {
        return sin(T1[1]);
    }
    static T sum_first_second(Tensor<T,1> T1)
    {
        return T1[0]+T1[1];
    }
    static T xsq_times_3(Tensor<T,1> T1)
    {
        return 3*T1[0]*T1[0];
    }
    static T x_times_4(Tensor<T,1> T1)
    {
        return 4*T1[0];
    }
    static T two(Tensor<T,1> T1)
    {
        return 2;
    }
};

int main()
{
    funcs_for_layer<double,3> funcs_for_f_layer_0;
    funcs_for_f_layer_0.A[0]=funcs<double>::first_elem;
    funcs_for_f_layer_0.A[1]=funcs<double>::second_elem;
    funcs_for_f_layer_0.A[2]=funcs<double>::prod_first_second;

    funcs_for_layer<double,3> funcs_for_f_layer_1;
    funcs_for_f_layer_1.A[0]=funcs<double>::log10_of_third;
    funcs_for_f_layer_1.A[1]=funcs<double>::sin_of_second;

    Fullyconnected<double> network_for_Autograd_f(dim_for_Autograd_f,std::vector<int>(2,dim_for_Autograd_f),std::vector<std::string>(2,"tanh"));
    network_for_Autograd_f.layers[0].free_funcs=funcs_for_f_layer_0.A;
    network_for_Autograd_f.layers[1].free_funcs=funcs_for_f_layer_1.A;

    network_for_Autograd_f.initialize_weights("identity");

    std::vector<Training_data<double>> training_data_for_Autograd_f;//x_1 =2, x_2=free, x_3=0 x_2 in [2,7]
    for(auto i=0;i<100;i++)
    {
        Tensor<double,1> input(std::vector<double>{2.0,2+5*i/100.f,0.0});
        double x[2]={input[0],input[1]};
        Tensor<double,1> target(std::vector<double>{f(x),0,0});
        Training_data<double> temp(input.getsize(),target.getsize());
        temp.input=input;
        temp.target=target;
        training_data_for_Autograd_f.push_back(temp);
    }
    if(0)
    for(auto i=0;i<100;i++)
    {
        network_for_Autograd_f.free_forward(training_data_for_Autograd_f[i].input);
        for(auto j=0;j<2;j++)
        {
            std::cout << "Layer " << j << " input: " << network_for_Autograd_f.layers[j].input << std::endl;
            std::cout << "Layer " << j << " output: " << network_for_Autograd_f.layers[j].output << std::endl;
        }
        //give out loss
        std::cout << "f(" << training_data_for_Autograd_f[i].input << ") = " << free_Loss_f<double,double>(network_for_Autograd_f.output_size,network_for_Autograd_f.layers[network_for_Autograd_f.layers.size()-1].output,training_data_for_Autograd_f[i].target) << std::endl;
        std::cout << "f(" << training_data_for_Autograd_f[i].input << ") = " << f(new double[2]{training_data_for_Autograd_f[i].input[0],training_data_for_Autograd_f[i].input[1]}) << std::endl;
        std::cin.get();
    }
    
    Optimizer<double> optimizer_for_Autograd_f(network_for_Autograd_f,training_data_for_Autograd_f,"free_Loss_f");
    
    if(func_name=='f')
    for(auto i=0;i<10;i++)
    {
        optimizer_for_Autograd_f.independent_optimization_step(training_data_for_Autograd_f[i]);
        //get independent_get_del_L_del_L_k
        std::vector<Tensor<double,1>> d_Loss_d_L=optimizer_for_Autograd_f.independent_get_del_L_del_L_k(training_data_for_Autograd_f[i]);
        std::cout << "\nInput: " << training_data_for_Autograd_f[i].input << std::endl;
        for(auto j=0;j<1;j++)
        {
            std::cout << "d_Loss_d_L[" << 0 << "]: " << d_Loss_d_L[1] << std::endl;
        }
        
        auto nabla=nabla_f(f,new double[2]{training_data_for_Autograd_f[i].input[0],training_data_for_Autograd_f[i].input[1]});
        std::vector<double> nabla_V{std::get<0>(nabla)[0],std::get<0>(nabla)[1]};
        Tensor<double,1> nabla_T(nabla_V);
        nabla_T=nabla_T*(-1);
        std::cout << "nabla_f: " << nabla_T << std::endl;
        optimizer_for_Autograd_f.reset_sum_changes_of_weights_and_biases();
        //stream both to file

    }


    funcs_for_layer<double,2> funcs_for_g_layer_0;
    funcs_for_g_layer_0.A[0]=funcs<double>::sum_first_second;
    funcs_for_g_layer_0.A[1]=funcs<double>::prod_first_second;

    Fullyconnected<double> network_for_Autograd_g(dim_for_Autograd_g,std::vector<int>(1,dim_for_Autograd_g),std::vector<std::string>(1,"tanh"));
    network_for_Autograd_g.layers[0].free_funcs=funcs_for_g_layer_0.A;

    network_for_Autograd_g.initialize_weights("identity");

    std::vector<Training_data<double>> training_data_for_Autograd_g;//x_1 =2, x_2=free, x_3=0 x_2 in [0,4]
    for(auto i=0;i<100;i++)
    {
        Tensor<double,1> input(std::vector<double>{2,4*i/100.f});
        double x[2]={input[0],input[1]};
        Tensor<double,1> target(std::vector<double>{g(x),0});
        Training_data<double> temp(input.getsize(),target.getsize());
        temp.input=input;
        temp.target=target;
        training_data_for_Autograd_g.push_back(temp);
    }
    
    if(0)
    for(auto i=0;i<100;i++)
    {
        network_for_Autograd_g.free_forward(training_data_for_Autograd_g[i].input);
        for(auto j=0;j<1;j++)
        {
            std::cout << "Layer " << j << " input: " << network_for_Autograd_g.layers[j].input << std::endl;
            std::cout << "Layer " << j << " output: " << network_for_Autograd_g.layers[j].output << std::endl;
        }
        //give out loss
        std::cout << "g(" << training_data_for_Autograd_g[i].input << ") = " << free_Loss_g<double,double>(network_for_Autograd_g.output_size,network_for_Autograd_g.layers[network_for_Autograd_g.layers.size()-1].output,training_data_for_Autograd_g[i].target) << std::endl;
        std::cout << "g(" << training_data_for_Autograd_g[i].input << ") = " << g(new double[2]{training_data_for_Autograd_g[i].input[0],training_data_for_Autograd_g[i].input[1]}) << std::endl;
        std::cin.get();
    }

    Optimizer<double> optimizer_for_Autograd_g(network_for_Autograd_g,training_data_for_Autograd_g,"free_Loss_g");

    if(func_name=='g')
    for(auto i=0;i<10;i++)
    {
        optimizer_for_Autograd_g.independent_optimization_step(training_data_for_Autograd_g[i]);
        //get independent_get_del_L_del_L_k
        std::vector<Tensor<double,1>> d_Loss_d_L=optimizer_for_Autograd_g.independent_get_del_L_del_L_k(training_data_for_Autograd_g[i]);
        std::cout << "\nInput: " << training_data_for_Autograd_g[i].input << std::endl;
        for(auto j=0;j<1;j++)
        {
            std::cout << "d_Loss_d_L[" << 1 << "]: " << d_Loss_d_L[1] << std::endl;
        }
        
        auto nabla=nabla_f(g,new double[2]{training_data_for_Autograd_g[i].input[0],training_data_for_Autograd_g[i].input[1]});
        std::vector<double> nabla_V{std::get<0>(nabla)[0],std::get<0>(nabla)[1]};
        Tensor<double,1> nabla_T(nabla_V);
        std::cout << "nabla_g: " << nabla_T << std::endl;
        optimizer_for_Autograd_g.reset_sum_changes_of_weights_and_biases();
    }

    funcs_for_layer<double,3> funcs_for_h_layer_0;
    funcs_for_h_layer_0.A[0]=funcs<double>::xsq_times_3;
    funcs_for_h_layer_0.A[1]=funcs<double>::x_times_4;
    funcs_for_h_layer_0.A[2]=funcs<double>::two;

    Fullyconnected<double> network_for_Autograd_h(dim_for_Autograd_h,std::vector<int>(1,dim_for_Autograd_h),std::vector<std::string>(1,"tanh"));
    network_for_Autograd_h.layers[0].free_funcs=funcs_for_h_layer_0.A;

    network_for_Autograd_h.initialize_weights("identity");

    std::vector<Training_data<double>> training_data_for_Autograd_h;//x_1 =2, x_2=free, x_3=0 x_2 in [0,4]
    for(auto i=0;i<100;i++)
    {
        Tensor<double,1> input(std::vector<double>{4*i/100.0,0,0});
        double x[2]={input[0],input[1]};
        Tensor<double,1> target(std::vector<double>{h(x),0,0});
        Training_data<double> temp(input.getsize(),target.getsize());
        temp.input=input;
        temp.target=target;
        training_data_for_Autograd_h.push_back(temp);
    }

    if(0)
    for(auto i=0;i<100;i++)
    {
        network_for_Autograd_h.free_forward(training_data_for_Autograd_h[i].input);
        for(auto j=0;j<1;j++)
        {
            std::cout << "Layer " << j << " input: " << network_for_Autograd_h.layers[j].input << std::endl;
            std::cout << "Layer " << j << " output: " << network_for_Autograd_h.layers[j].output << std::endl;
        }
        //give out loss
        std::cout << "h(" << training_data_for_Autograd_h[i].input << ") = " << free_Loss_h<double,double>(network_for_Autograd_h.output_size,network_for_Autograd_h.layers[network_for_Autograd_h.layers.size()-1].output,training_data_for_Autograd_h[i].target) << std::endl;
        std::cout << "h'(" << training_data_for_Autograd_h[i].input << ") = " << h(new double[2]{training_data_for_Autograd_h[i].input[0],training_data_for_Autograd_h[i].input[1]}) << std::endl;
        std::cin.get();
    }

    Optimizer<double> optimizer_for_Autograd_h(network_for_Autograd_h,training_data_for_Autograd_h,"free_Loss_h");


    if(func_name=='h')
    for(auto i=0;i<100;i++)
    {
        optimizer_for_Autograd_h.independent_optimization_step(training_data_for_Autograd_h[i]);
        //get independent_get_del_L_del_L_k
        std::vector<Tensor<double,1>> d_Loss_d_L=optimizer_for_Autograd_h.independent_get_del_L_del_L_k(training_data_for_Autograd_h[i]);
        std::cout << "\nInput: " << training_data_for_Autograd_h[i].input << std::endl;
        for(auto j=0;j<1;j++)
        {
            std::cout << "d_Loss_d_L[" << 0 << "]: " << d_Loss_d_L[1] << std::endl;
        }
        
        auto nabla=nabla_f(h,new double[2]{training_data_for_Autograd_h[i].input[0],training_data_for_Autograd_h[i].input[1]});
        std::vector<double> nabla_V{std::get<0>(nabla)[0],std::get<0>(nabla)[1]};
        Tensor<double,1> nabla_T(nabla_V);
        std::cout << "nabla_h: " << nabla_T << std::endl;
        optimizer_for_Autograd_h.reset_sum_changes_of_weights_and_biases();
    }




    return 0;
}










