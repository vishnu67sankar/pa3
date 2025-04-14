#include <vector>
#include <numeric> // For std::iota
#include <vector>
#include <mpi.h>
#include <cmath> // For std::ceil
#include <cassert> // For assert

// Assuming functions.h includes the necessary pair definitions or they are defined elsewhere
// typedef std::pair<int, int> IntPair;
// typedef std::pair<IntPair, int> MatrixEntry; // Example: { {row, col}, value }

void distribute_matrix_2d(int m, int n, std::vector<std::pair<std::pair<int, int>, int>> &full_matrix,
                          std::vector<std::pair<std::pair<int, int>, int>> &local_matrix,
                          int root, MPI_Comm comm_2d)
{
    int world_rank, world_size;
    MPI_Comm_rank(comm_2d, &world_rank);
    MPI_Comm_size(comm_2d, &world_size);

    int dims[2], periods[2], coords[2];
    MPI_Cart_get(comm_2d, 2, dims, periods, coords); // Get dimensions and coordinates
    int proc_rows = dims[0];
    int proc_cols = dims[1];
    assert(proc_rows * proc_cols == world_size); // Ensure grid dimensions match communicator size

    // --- Determine row/column ranges for each processor block ---
    // This handles non-divisible cases by distributing remainder rows/cols
    std::vector<int> row_starts(proc_rows + 1);
    std::vector<int> col_starts(proc_cols + 1);

    for (int i = 0; i <= proc_rows; ++i) {
        row_starts[i] = (i * m) / proc_rows;
    }
     for (int j = 0; j <= proc_cols; ++j) {
        col_starts[j] = (j * n) / proc_cols;
    }

    // --- Root Process Logic ---
    if (world_rank == root)
    {
        // Create temporary storage for each process
        std::vector<std::vector<int>> send_rows(world_size);
        std::vector<std::vector<int>> send_cols(world_size);
        std::vector<std::vector<int>> send_vals(world_size);

        // Iterate through the full matrix *once*
        for (const auto &entry : full_matrix)
        {
            int row = entry.first.first;
            int col = entry.first.second;
            int val = entry.second;

            // Find which processor block this entry belongs to
            int target_proc_row = -1;
            int target_proc_col = -1;

            // Find target processor row
             for (int pr = 0; pr < proc_rows; ++pr) {
                if (row >= row_starts[pr] && row < row_starts[pr+1]) {
                    target_proc_row = pr;
                    break;
                }
            }
             // Find target processor column
            for (int pc = 0; pc < proc_cols; ++pc) {
                if (col >= col_starts[pc] && col < col_starts[pc+1]) {
                    target_proc_col = pc;
                    break;
                }
            }

            // Should always find a target block if logic is correct
            assert(target_proc_row != -1 && target_proc_col != -1); 

            // Get the rank of the target processor
            int target_coords[2] = {target_proc_row, target_proc_col};
            int target_rank;
            MPI_Cart_rank(comm_2d, target_coords, &target_rank);

            // Add the serialized entry to the target rank's buffers
            send_rows[target_rank].push_back(row);
            send_cols[target_rank].push_back(col);
            send_vals[target_rank].push_back(val);
        }

        // Send data to all processes (including potentially self)
        for (int target_rank = 0; target_rank < world_size; ++target_rank)
        {
            int count = send_rows[target_rank].size(); // Number of entries for this rank

            if (target_rank == root) {
                // Optimization: If sending to self, just copy directly
                local_matrix.clear();
                local_matrix.reserve(count);
                for(int i = 0; i < count; ++i) {
                    local_matrix.push_back({{send_rows[root][i], send_cols[root][i]}, send_vals[root][i]});
                }
            } else {
                // 1. Send the count
                MPI_Send(&count, 1, MPI_INT, target_rank, 0, comm_2d);
                if (count > 0) {
                     // 2. Send row indices
                    MPI_Send(send_rows[target_rank].data(), count, MPI_INT, target_rank, 1, comm_2d);
                    // 3. Send column indices
                    MPI_Send(send_cols[target_rank].data(), count, MPI_INT, target_rank, 2, comm_2d);
                    // 4. Send values
                    MPI_Send(send_vals[target_rank].data(), count, MPI_INT, target_rank, 3, comm_2d);
                }
            }
        }
    }
    // --- Non-Root Process Logic ---
    else
    {
        int count;
        MPI_Status status;

        // 1. Receive the count
        MPI_Recv(&count, 1, MPI_INT, root, 0, comm_2d, &status);

        if (count > 0) {
            // Allocate space
            std::vector<int> recv_rows(count);
            std::vector<int> recv_cols(count);
            std::vector<int> recv_vals(count);

            // 2. Receive row indices
            MPI_Recv(recv_rows.data(), count, MPI_INT, root, 1, comm_2d, &status);
            // 3. Receive column indices
            MPI_Recv(recv_cols.data(), count, MPI_INT, root, 2, comm_2d, &status);
            // 4. Receive values
            MPI_Recv(recv_vals.data(), count, MPI_INT, root, 3, comm_2d, &status);

            // Reconstruct the local matrix
            local_matrix.clear();
            local_matrix.reserve(count);
            for (int i = 0; i < count; ++i) {
                local_matrix.push_back({{recv_rows[i], recv_cols[i]}, recv_vals[i]});
            }
        } else {
            // If count is 0, ensure local_matrix is empty
            local_matrix.clear();
        }
    }
}