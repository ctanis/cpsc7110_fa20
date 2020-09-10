#include <iostream>
#include <iomanip>

#include <Kokkos_Core.hpp>


const long int SIZE=1000000;

int main(int argc, char *argv[])
{

    Kokkos::initialize(argc, argv);
    {
        std::cout << "This code will execute here: " <<
            Kokkos::DefaultExecutionSpace::name() << std::endl;

        Kokkos::View<double*> data("my data", SIZE);

        Kokkos::parallel_for(SIZE, KOKKOS_LAMBDA(const int i) {
                data(i)=i;
            });

        double global_sum =0;
        Kokkos::parallel_reduce(SIZE, KOKKOS_LAMBDA(const int i, double& sum) {
                sum += data(i);
            }, global_sum);
            

        std::cout << "analytic sum: " << SIZE*(SIZE-1)/2 << std::endl;
        std::cout << "kokkos reduction: " <<
            std::setprecision(std::numeric_limits<long double>::digits10 + 1) <<
            global_sum << std::endl;

    }
    Kokkos::finalize();
    return 0;
}
