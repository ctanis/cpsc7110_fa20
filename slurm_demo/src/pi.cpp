#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <unistd.h>
#include <ctime>

#define COUNT 10000


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, np;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);



    srandom(getpid() ^ time(NULL));
    
    int hits=0;
    for (int i=0; i<COUNT; i++)
    {
        double x = random() * 1.0 / RAND_MAX;
        double y = random() * 1.0 / RAND_MAX;

        if (sqrt(x*x+y*y) < 1)
        {
            hits++;
        }
    }

    //printf("%lf\n", hits*4.0/COUNT);

    // MPI_Status status;
    // if (rank == 0)
    // {
    //     for (int r=1; r<np; r++)
    //     {
    //         printf("receiving from %d\n", r);
    //         int other;
    //         MPI_Recv(&other, 1, MPI_INT, r, 0, MPI_COMM_WORLD, &status);
    //         hits += other;
    //     }

    //     printf("%lf = 4.0* %d/%d\n", hits*4.0/(np*COUNT), hits, np*COUNT);
    // }
    // else
    // {
    //     printf("sending from %d\n", rank);
    //     MPI_Send(&hits, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    // }

    int total;
    MPI_Reduce(&hits,&total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank==0)
    {
        printf("%lf = 4.0* %d/%d\n", total*4.0/(np*COUNT), total, np*COUNT);
    }
    // else
    // {
    //     printf("not on rank 0: %d\n", total);
    // }


    MPI_Finalize();
    return 0;
}
