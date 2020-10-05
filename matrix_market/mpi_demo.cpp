#include <iostream>
#include <fstream>
#include <mpi.h>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>


void read_matrix_market(std::string filename,
                        std::vector<int>& ia,
                        std::vector<int>& ja,
                        std::vector<double>& data);
#define LOG(x) std::cerr << x << std::endl;


template <typename T>
std::string stringify(T v)    {
    std::ostringstream o;
    o << v;
    return o.str();
}



int owner(int gid, std::vector<int> partition)
{
    int np=partition.size();
        
    // binary search of partition
    int low=0;
    int high=np;
    int mid;

    do
    {
        mid = (low+high) / 2;

        if (gid < partition[mid])
        {
            high = mid;

        }
        else if (gid >= partition[mid+1])
        {
            low = mid+1;
        }
        else
        {
            return mid;
        }
    }
    while (low < high);

    throw std::runtime_error("unknown global " + stringify(gid));
    return -1;
}



int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, np;
    char machine_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    std::ofstream* outp(nullptr);


    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(machine_name, &name_len);

    if (np > 0 && rank > 0)     // dump non root processor output to files
    {
        // for parallel runs, rebind stdout to a unique file
        outp=new std::ofstream("STDERR-"+stringify(rank)+".out");
        outp->exceptions(std::ofstream::badbit|std::ofstream::failbit);
        std::cerr.rdbuf(outp->rdbuf());
    }


    LOG("Hello from " << rank <<  " of " << np << " on " << machine_name);

    // for simplicity, just read the matrix on every participating process ...
    std::vector<int> ia;
    std::vector<int> ja;
    std::vector<double> data;

    read_matrix_market(argv[1], ia, ja, data);
    int nrows = ia.size()-1;

    // a crude partitioning:
    // after this logic, the current rank owns rows [first_row, first_row+local_count)

    int local_count=nrows/np;
    int first_row=local_count*rank;
    int mod=nrows%np;
    if (rank < mod)
    {
        local_count++;
        first_row += rank;
    }
    else
    {
        first_row += mod;
    }
    

    LOG("owns rows " << first_row << " - " << first_row+local_count-1<< " of " << nrows);
    std::vector<int> partition(np);
    MPI_Allgather(&first_row, 1, MPI_INT, partition.data(), 1, MPI_INT, MPI_COMM_WORLD);
    partition.push_back(nrows); // similar to ia-- last element contains global size

    // At this point, every processor knows unambigously what range of global
    // row ids is owned by every participating rank.  This is not the only way
    // to compute this, but this way is very general -- clever partition
    // algorithms could require each processor to make local independent
    // decisions on where they should start.  This gather ensures that all
    // processors are in agreement.


    // collapse the CSR into a local problem
    std::vector<int> local_ia;
    std::vector<int> local_ja;
    std::vector<double> local_data;
    int local_nnz = ia[first_row+local_count]-ia[first_row];

    local_ia.resize(local_count+1);
    local_ja.resize(local_nnz);
    local_data.resize(local_nnz);

    std::map<int, int> g2l;     // global-to-local index map for nonlocal indices
    for (unsigned int i=0; i<local_ia.size(); i++)
    {
        local_ia[i]=ia[first_row+i]-ia[first_row];
    }

    int first_unused=local_count;
    for (unsigned int c=0; c<local_ja.size(); c++)
    {
        int index=ia[first_row]+c;
        int col=ja[index];
        
        if (col < first_row || col >= first_row+local_count)
        {
            // for columns that do not correspond to rows owned by this rank,
            // we need to augment our local problem with new indices that
            // "map" to the off-rank global indices.  The first step here is
            // to figure out exactly what those indices are.

            auto it=g2l.find(col);
            if (it != g2l.end())
            {
                col=it->second;
            }
            else
            {
                g2l[col]=first_unused;
                col=first_unused;
                first_unused++;
            }
        }
        else
        {
            col -= first_row;
        }

        local_ja[c] = col;
        local_data[c]=data[index];
    }
    

    // "throw away" global data ..
    ia.resize(0);
    ja.resize(0);
    data.resize(0);

    // -----------------------------------------
    // At this point, we pretend that the global initialization did not need
    // to occur and we imagine the matrix problem truly distributed across
    // ranks.  In other words, the local problem starts here....

#ifdef PRINT_CRS
    
    LOG("ia: " << local_ia.size());
    for (unsigned int i=0; i<local_ia.size(); i++)
    {
        LOG(i << ": " << local_ia[i]);
    }

    LOG("ja: " << local_ja.size());
    for (unsigned int i=0; i<local_ja.size(); i++)
    {
        LOG(i << ": " << local_ja[i]);
    }

    LOG("data: " << local_data.size());
    for (unsigned int i=0; i<local_data.size(); i++)
    {
        LOG(i << ": " << local_data[i]);
    }

    LOG("off diagonal columns: ");
    for (auto it=g2l.begin(); it != g2l.end(); it++)
    {
        LOG(it->first << " -> " << it->second );
    }

#endif

    // g2l tells us where locally we are going to store non-local data when it
    // comes time to matrix/vector multiply
    // let's make sure that all other ranks know what we'll need from them when the time comes.

    // The goal of this section of code is to build up vectors for building
    // messags containing local data, destined for other ranks, and for
    // unpacking incoming messages


    // Identify owner of all off-rank indices, so we can request that data
    // from the appropriate ranks
    std::vector<std::set<int> > initial_request_map(np);
    for (auto it=g2l.begin(); it!= g2l.end(); it++)
    {
        int global=it->first;

        // figure out the owning rank of global by looking at partition
        int orank=owner(global, partition);

        // request_map is a set of global ids needed from each other rank
        initial_request_map[orank].insert(global);
    }
    


    // after this initial map building, the following will hold:
    // outgoing_counts contains the number of items to send to each other rank
    // outgoing_ids[np] contains the particular local indices bound for each rank
    // incoming_counts contains the number of items we will receive from each other rank
    // incoming_ids[n[ contains the local indices for all incoming data;
    std::vector<int> outgoing_counts(np);
    std::vector<std::vector<int> > outgoing_ids(np);
    std::vector<int> incoming_counts(np);
    std::vector<std::vector<int> > incoming_ids(np);

    for (unsigned int r=0; r<np; r++)
    {
        incoming_counts[r] = initial_request_map[r].size();
    }


    MPI_Alltoall(incoming_counts.data(), 1, MPI_INT,
                 outgoing_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    LOG("sending data requests:");
    for (int i=0; i<np; i++)
    {
        if (i == rank)
        {
            LOG("rank " << i << " is me -- this should be 0: " << incoming_counts[i]);
        }
        else
        {
            LOG("rank " << i << " needs to send " << incoming_counts[i] << " to me");
        }
        
    }
      

    LOG("received data requests:");
    for (int i=0; i<np; i++)
    {
        if (i == rank)
        {
            LOG("rank " << i << " is me -- this should be 0: " << outgoing_counts[i]);
        }
        else
        {
            LOG("rank " << i << " needs " << outgoing_counts[i] << " from me");
        }
        
    }

    // build the incoming/outcoing
        
    std::list<MPI_Request> reqs;
    for (int i=0; i<np; i++)
    {
        if (incoming_counts[i] != 0)
        {
            incoming_ids[i].resize(incoming_counts[i]);
            int idx=0;

            for (auto it=initial_request_map[i].begin(); it != initial_request_map[i].end(); it++)
            {
                incoming_ids[i][idx]=*it;
                idx++;
            }
            
            reqs.push_back(MPI_Request());
            MPI_Isend(incoming_ids[i].data(), incoming_counts[i], MPI_INT, i, 0,
                      MPI_COMM_WORLD, &reqs.back());

        }

        if (outgoing_counts[i] != 0)
        {
            outgoing_ids[i].resize(outgoing_counts[i]);
            reqs.push_back(MPI_Request());

            MPI_Irecv(outgoing_ids[i].data(), outgoing_counts[i], MPI_INT, i, 0,
                      MPI_COMM_WORLD, &reqs.back());
            
            
        }

    }

    // wait for all these requests
    for (auto it=reqs.begin(); it != reqs.end(); it++)
    {
        MPI_Wait(&(*it), NULL);
    }
    reqs.resize(0);

    // transform all maps into local versions

    for (int r=0; r<np; r++)
    {
        // all maps should be transformed into local indices
        for (unsigned int i=0; i<incoming_ids[r].size(); i++)
        {
            incoming_ids[r][i] = g2l.at(incoming_ids[r][i]);
            // LOG("incoming " << incoming_ids[r][i]);
        }

        for (unsigned int i=0; i<outgoing_ids[r].size(); i++)
        {
            outgoing_ids[r][i] -= first_row;
            // LOG("outgoing " << outgoing_ids[r][i]);
        }
    }


    // so what does this do for us?  Imagine that we are going to do a sparse
    // matrix vector multiplication, and the vector we are multiplying by are
    // distributed over the participating processes

    // We have a local vector for multiplying the local matrix.  this vector
    // has nrows elements (because that's what this rank owns) PLUS an entry
    // for all off-diagonal nonzero columns

    std::vector<double> vectorB(local_count+g2l.size());

    // let's initialize this with numbers indicating the nature of the data
    // notice how we are not changing the "ghost" values that belong to other
    // ranks
    for (int i=0; i<local_count; i++)
    {
        if (rank != 0)
        {
            vectorB[i]=(i + first_row) + 1.0/(rank * 1000); // something like 2.00rank
        }
        else
        {
            vectorB[i]=i;
        }
        
    }


    // before we can continue, we must make sure we have a local copy of the
    // vectorB values that really exist on other processes.. this is what our
    // maps are for.

    {

        std::vector<std::vector<double> > outgoing_data(np);
        std::vector<std::vector<double> > incoming_data(np);
    
        for (int r=0; r<np; r++)
        {
            if (incoming_counts[r] != 0)
            {
                incoming_data[r].resize(incoming_counts[r]);

                reqs.push_back(MPI_Request());
                MPI_Irecv(incoming_data[r].data(), incoming_counts[r], MPI_DOUBLE, r, 0,
                          MPI_COMM_WORLD, &reqs.back());
            }


            if (outgoing_counts[r] != 0)
            {
                outgoing_data[r].resize(outgoing_counts[r]);

                for (unsigned int z=0; z<outgoing_counts[r]; z++)
                {
                    outgoing_data[r][z] = vectorB[outgoing_ids[r][z]];
                }

                reqs.push_back(MPI_Request());
                MPI_Isend(outgoing_data[r].data(), outgoing_counts[r], MPI_DOUBLE, r, 0,
                          MPI_COMM_WORLD, &reqs.back());

            }

        }

        // wait for all these requests
        for (auto it=reqs.begin(); it != reqs.end(); it++)
        {
            MPI_Wait(&(*it), NULL);
        }
        reqs.resize(0);

        // copy incoming data into vectorB

        for (int r=0; r<np; r++)
        {
            if (incoming_counts[r] != 0)
            {
                for (unsigned int z=0; z<incoming_ids[r].size(); z++)
                {
                    vectorB[incoming_ids[r][z]] = incoming_data[r][z];
                }
            }
        }


        // see how the local values in vectorB are populated with the valuers
        // from the other ranks:

        LOG("final vectorB");
        for (unsigned int i=0; i<vectorB.size(); i++)
        {
            if (i == local_count)
            {
                LOG("---- off-diagonals below ----");
            }
            LOG("i: " << vectorB[i]);
        }

    }


    
    
    

    if (outp != nullptr)
    {
        delete outp;
        outp=nullptr;
    }

    MPI_Finalize();
    return 0;
}

// Local Variables:
// compile-command: "OMPI_CXX=g++-10 && mpic++ -Wall -std=c++11 mpi_demo.cpp matrix_market.cpp -o mpi_demo"
// End:
