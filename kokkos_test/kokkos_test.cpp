#include <iostream>
#include <Kokkos_Core.hpp>

const int SIZE=1000000;

int main(int argc, char *argv[])
{
    Kokkos::initialize(argc, argv);

    {

        //std::vector<int> data(SIZE, 0);

        Kokkos::View<double> data("my data", SIZE);

        Kokkos::parallel_for(SIZE,
                             KOKKOS_LAMBDA(const int i)
                             {
                                 data(i)=i;
                             });
    
    }
    
    Kokkos::finalize();
    return 0;
}
