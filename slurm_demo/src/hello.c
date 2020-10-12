#include <mpi.h>
#include <stdio.h>


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, np, len;
    char name[MPI_MAX_PROCESSOR_NAME];

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Get_processor_name(name, &len);

    printf("%5d/%d - %s\n", rank, np, name);
    

    MPI_Finalize();
    return 0;
}
