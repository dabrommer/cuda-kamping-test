#include <cuda_runtime.h>
#include <iostream>

#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/p2p/recv.hpp"
#include "kamping/p2p/send.hpp"

#define N 10

int MPI_test(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    float* d_buf;
    cudaMalloc((void**)&d_buf, N * sizeof(float));

    if (rank == 0) {
        float h_data[N];
        for (int i = 0; i < N; ++i) h_data[i] = rank + 1;
        cudaMemcpy(d_buf, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
		MPI_Send(d_buf, N, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);

    } else if (rank == 1) {
        MPI_Recv(d_buf, N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        float h_result[N];
        cudaMemcpy(h_result, d_buf, N * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "Rank 1 received: ";
        for (int i = 0; i < N; ++i) std::cout << h_result[i] << " ";
        std::cout << std::endl;
    }

    cudaFree(d_buf);
	MPI_Barrier(MPI_COMM_WORLD);

    return 0;

}

int kamping_test() {

    kamping::Environment  e;
    kamping::Communicator comm;

	int rank = comm.rank();

    float* d_buf;
    cudaMalloc((void**)&d_buf, N * sizeof(float));

    if (rank == 0) {
        float h_data[N];
        for (int i = 0; i < N; ++i) h_data[i] = rank + 1;
        cudaMemcpy(d_buf, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

        comm.send(kamping::send_buf(d_buf), kamping::destination(1));

    } else if (rank == 1) {
        comm.recv(kamping::recv_buf<kamping::no_resize>(d_buf));

        float h_result[N];
        cudaMemcpy(h_result, d_buf, N * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "Rank 1 received: ";
        for (int i = 0; i < N; ++i) std::cout << h_result[i] << " ";
        std::cout << std::endl;
    }

    cudaFree(d_buf);
    return 0;
}
 
int main(int argc, char **argv) {
    MPI_test(argc, argv);
	kamping_test();
}



