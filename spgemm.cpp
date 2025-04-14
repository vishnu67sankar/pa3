#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <iostream>
#include <mpi.h>
#include <cassert>
#include "functions.h" 
#include <limits>


void broadcast_block(std::vector<std::pair<std::pair<int, int>, int>>& block,
                     int root_rank_in_comm, MPI_Comm comm) //helper function to serialize and broadcast a block
{
    int my_rank_in_comm;
    MPI_Comm_rank(comm, &my_rank_in_comm);

    int count = 0;
    std::vector<int> rows, cols, vals;

    if (my_rank_in_comm == root_rank_in_comm) {
        count = block.size();
        rows.reserve(count);
        cols.reserve(count);
        vals.reserve(count);
        for (const auto& entry : block) {
            rows.push_back(entry.first.first);
            cols.push_back(entry.first.second);
            vals.push_back(entry.second);
        }
    }

    
    MPI_Bcast(&count, 1, MPI_INT, root_rank_in_comm, comm);

    
    if (my_rank_in_comm != root_rank_in_comm) 
    { 
        rows.resize(count);
        cols.resize(count);
        vals.resize(count);
        block.clear(); 
        block.reserve(count);
    }

    if (count > 0) 
    {
        
        MPI_Bcast(rows.data(), count, MPI_INT, root_rank_in_comm, comm);
        MPI_Bcast(cols.data(), count, MPI_INT, root_rank_in_comm, comm);
        MPI_Bcast(vals.data(), count, MPI_INT, root_rank_in_comm, comm);
    }

    
    if (my_rank_in_comm != root_rank_in_comm && count > 0) {
         for (int i = 0; i < count; ++i) {
            block.push_back({{rows[i], cols[i]}, vals[i]});
        }
    }
}


void spgemm_2d(int m, int p, int n,
               std::vector<std::pair<std::pair<int, int>, int>>& A, 
               std::vector<std::pair<std::pair<int, int>, int>>& B, 
               std::vector<std::pair<std::pair<int, int>, int>>& C, 
               std::function<int(int, int)> plus,
               std::function<int(int, int)> times,
               MPI_Comm row_comm, MPI_Comm col_comm)
{
    int my_rank_in_row, my_rank_in_col; 
    MPI_Comm_rank(row_comm, &my_rank_in_row); 
    MPI_Comm_rank(col_comm, &my_rank_in_col); 

    int proc_dim_row, proc_dim_col;
    MPI_Comm_size(row_comm, &proc_dim_row); 
    MPI_Comm_size(col_comm, &proc_dim_col); 
    //assert(proc_dim_row == proc_dim_col); 
    int proc_dim = proc_dim_row; 

    const int infinity = std::numeric_limits<int>::max();
    std::map<std::pair<int, int>, int> C_map; //am using a map to store results for the local C block

    std::vector<std::pair<std::pair<int, int>, int>> A_broadcast_block;
    std::vector<std::pair<std::pair<int, int>, int>> B_broadcast_block;

    for (int k = 0; k < proc_dim; ++k) //this is where sparse SUMMA algorithm outer loop begins
    {
        A_broadcast_block = (my_rank_in_row == k) ? A : std::vector<std::pair<std::pair<int, int>, int>>();
        broadcast_block(A_broadcast_block, k, row_comm);
        B_broadcast_block = (my_rank_in_col == k) ? B : std::vector<std::pair<std::pair<int, int>, int>>();
        broadcast_block(B_broadcast_block, k, col_comm); //using col_comm for B broadcast as given in the pseudo code

        if (A_broadcast_block.empty() || B_broadcast_block.empty()) //if any of the received block is empty just skip the computation 
        {
            continue;
        }

        std::map<int, std::vector<std::pair<int, int>>> B_indexed_by_row;
        for (const auto& entry : B_broadcast_block) {
            int B_row = entry.first.first; 
            int B_col = entry.first.second;
            int B_val = entry.second;
            B_indexed_by_row[B_row].push_back({B_col, B_val});
        }

        for (const auto& A_entry : A_broadcast_block) {
            int A_row = A_entry.first.first;    
            int A_col = A_entry.first.second;
            int A_val = A_entry.second;

            auto it_B_rows = B_indexed_by_row.find(A_col);
            if (it_B_rows != B_indexed_by_row.end()) 
            {
                for (const auto& B_col_val_pair : it_B_rows->second) {
                    int B_col = B_col_val_pair.first; 
                    int B_val = B_col_val_pair.second;
                    int product = times(A_val, B_val); //find product using times

                    if (product == infinity) 
                    {
                        continue;
                    }

                    std::pair<int, int> C_coord = {A_row, B_col}; //find coordinate in C block

                    auto it_C = C_map.find(C_coord);
                    if (it_C != C_map.end()) 
                    {
                        it_C->second = plus(it_C->second, product);
                    } 
                    
                    else 
                    {
                         C_map[C_coord] = product;
                    }
                }
            }
        }
    }

    C.clear();
    C.reserve(C_map.size());
    for (const auto& entry : C_map) 
    {
        const int ADDITIVE_IDENTITY = 0;
        if (entry.second != ADDITIVE_IDENTITY) {
            C.push_back({entry.first, entry.second});
        }
    }
}
