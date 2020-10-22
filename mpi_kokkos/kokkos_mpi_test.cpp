#include <iostream>
#include <iomanip>

#include <Kokkos_Core.hpp>
#include <mpi.h>



#define DO_DOT_PRODUCT


const long int SIZE=1000000;

KOKKOS_INLINE_FUNCTION
double fuse(double a, double b)
{
    return a*b;
}


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int np, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    std::cout << "mpi rank " << rank << " of " << np << std::endl;

    Kokkos::initialize(argc, argv);
    {
        std::cout << "This code will execute here: " <<
            Kokkos::DefaultExecutionSpace::name() << std::endl;

        Kokkos::View<double*> data("my data", SIZE);

        Kokkos::parallel_for("init1", SIZE, KOKKOS_LAMBDA(const int i) {
                data(i)=i;
            });

        double global_sum =0;
        Kokkos::parallel_reduce("reduce1", SIZE, KOKKOS_LAMBDA(const int i, double& sum) {
                sum += data(i);
            }, global_sum);
            

        std::cout << "analytic sum: " << SIZE*(SIZE-1)/2 << std::endl;
        std::cout << "kokkos reduction: " <<
            std::setprecision(std::numeric_limits<long double>::digits10 + 1) <<
            global_sum << std::endl;

#ifdef DO_DOT_PRODUCT
        Kokkos::View<double*> data2("my data", SIZE);
        double scale=10;
        Kokkos::parallel_for("init2", SIZE, KOKKOS_LAMBDA(const int i) {
                data2(i)=scale;
            });
        
        double scaled_dot=0;
        Kokkos::parallel_reduce("reduce2", SIZE, KOKKOS_LAMBDA(const int i, double& dot) {
                // dot += data(i)*data2(i);
                dot += fuse(data(i),data2(i));
            }, scaled_dot);

        std::cout << "dot product: " << scaled_dot << std::endl;

#endif        
        

    }
    
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
