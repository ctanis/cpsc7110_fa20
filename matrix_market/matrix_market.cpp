#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <stdexcept>



#define LINE_LEN 1024
#define LOG(x) std::cerr << x << std::endl;


void read_matrix_market(std::string filename,
                        std::vector<int>& ia,
                        std::vector<int>& ja,
                        std::vector<double>& data)
{
    std::fstream fin;
    fin.exceptions(std::ifstream::badbit);

    fin.open(filename);
    if (fin.fail())
    {
        throw std::runtime_error("cannot open file: " + filename);
    }
    

    std::vector<char> linep(LINE_LEN);
    std::stringstream sbuf;

    char* line=linep.data();

    int nrows, ncols, nnz;
    bool skip=true;

    std::vector< std::map<int, double> > build;

    // eat comments and load  dimensions
    while (skip)
    {
        fin.getline(line, LINE_LEN);

        if (line[0] != '%')
        {
            sbuf=std::stringstream(line);
            sbuf >> nrows >> ncols >> nnz;
            skip=false;
        }
    }

    LOG("got sizes: " << nrows << ", " << ncols << ", " << nnz);
    build.resize(nrows);

    while (fin.getline(line, LINE_LEN))
    {
        int row, col;
        double  v;

        if (line[0] != 0)
        {
            sbuf=std::stringstream(line);
            sbuf >> row >> col >> v;
            row--;                  // convert 1-based to 0-based
            col--;
            build[row][col]=v;
        }
    }
    

    LOG("finished building map")
    fin.close();

    // flatten into vectors
    ia.resize(nrows+1);
    ja.reserve(nnz);
    data.reserve(nnz);

    for (unsigned int i=0; i<build.size(); i++)
    {
        ia[i] = ja.size();

        for (auto it=build[i].begin(); it != build[i].end(); it++)
        {
            ja.push_back(it->first);
            data.push_back(it->second);
        }
    }
    ia[build.size()]=ja.size();
    
    return;
}


void print_row(int row, const std::vector<int>& ia,
               const std::vector<int>& ja,
               const std::vector<double>& data)
{
    for (int c=ia[row]; c<ia[row+1]; c++)
    {
        LOG("Row: " << row << "; Col: " << ja[c] << " : " << data[c]);
    }
}


#ifdef TEST_MM_READER

int main(int argc, char *argv[])
{
    std::vector<int> ia, ja;
    std::vector<double> data;

    try
    {
        read_matrix_market(argv[1], ia, ja, data);
        LOG(ia.size() << " == " << ja.size() << " == " << data.size());

        LOG("first row: ");
        print_row(0, ia, ja, data);

        LOG("random row in the middle: ");
        print_row(ia.size()/2, ia, ja, data);

        LOG("last row: ");
        print_row(ia.size()-2, ia, ja, data);
        

    }
    catch(const std::exception& e)
    {
        LOG("got an exception: " << e.what());
    }
    
    return 0;
}
#endif


// Local Variables:
// compile-command: "g++ -O3 -Wall matrix_market.cpp"
// End:
