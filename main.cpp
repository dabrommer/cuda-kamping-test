#include <cuda_runtime.h>
#include <iostream>
#include <span>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <kassert/kassert.hpp>

#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/p2p/recv.hpp"
#include "kamping/p2p/send.hpp"

#define N 10
#define X 42

void print_and_test(const char* test_name, const float* h_result) {
    std::cout << test_name << " Rank 1 received: ";
    for (int i = 0; i < N; ++i) std::cout << h_result[i] << " ";
    std::cout << std::endl;

    for (int i = 0; i < N; ++i) {
        KASSERT(h_result[i] == X, "Error in " << test_name);
    }
}

void mpi_test_raw_pointer(int rank) {
    float* d_buf;
    cudaMalloc((void**)&d_buf, N * sizeof(float));

    if (rank == 0) {
        float h_data[N];
        for (int i = 0; i < N; ++i) h_data[i] = X;
        cudaMemcpy(d_buf, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
		MPI_Send(d_buf, N, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Recv(d_buf, N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        float h_result[N];
        cudaMemcpy(h_result, d_buf, N * sizeof(float), cudaMemcpyDeviceToHost);
        print_and_test("MPI raw pointer", h_result);
    }
    cudaFree(d_buf);
}

void kamping_test_span(int rank) {
    kamping::Communicator comm;
    float* d_buf;
    cudaMalloc((void**)&d_buf, N * sizeof(float));
    std::span<float> d_span(d_buf, N);

    if (rank == 0) {
        float h_data[N];
        for (int i = 0; i < N; ++i) h_data[i] = X;
        cudaMemcpy(d_buf, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
        comm.send(kamping::send_buf(d_span), kamping::destination(1));

    } else if (rank == 1) {
        comm.recv(kamping::recv_buf<kamping::no_resize>(d_span));

        float h_result[N];
        cudaMemcpy(h_result, d_buf, N * sizeof(float), cudaMemcpyDeviceToHost);
        print_and_test("Kamping span", h_result);
    }
    cudaFree(d_buf);
}

void kamping_test_thrust_device(int rank) {
    kamping::Communicator comm;
    if (rank == 0) {
        thrust::host_vector<float> h_vec(N, X);
        comm.send(kamping::send_buf(h_vec), kamping::destination(1));

    } else if (rank == 1) {
        thrust::host_vector<float> h_vec(N, 99);
        comm.recv(kamping::recv_buf<kamping::no_resize>(h_vec));

        //thrust::host_vector<int> h_vec(d_vec.begin(), d_vec.end());
        print_and_test("Kamping thrust host vec", h_vec.data());
    }
}
 
void mpi_test_thrust_device(int rank) {
    if (rank == 0) {
        thrust::host_vector<float> d_vec(N, X);
        MPI_Send(d_vec.data(), N, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        thrust::host_vector<float> d_vec(N, 99);
        MPI_Recv(d_vec.data(), N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        thrust::host_vector<float> h_vec(d_vec.begin(), d_vec.end());
        print_and_test("MPI thrust host vec", h_vec.data());
    }
}

void kamping_test_raw_pointer(int rank) {
    kamping::Communicator comm;

    float* d_buf;
    cudaMalloc((void**)&d_buf, N * sizeof(float));

    if (rank == 0) {
        float h_data[N];
        for (int i = 0; i < N; ++i) h_data[i] = X;
        cudaMemcpy(d_buf, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
        comm.send(kamping::send_buf(d_buf), kamping::destination(1), kamping::send_count(N), kamping::send_type(MPI_FLOAT));
    } else if (rank == 1) {
        comm.recv(kamping::recv_buf<kamping::no_resize>(d_buf), kamping::recv_count(N), kamping::recv_type(MPI_FLOAT));
        float h_result[N];
        cudaMemcpy(h_result, d_buf, N * sizeof(float), cudaMemcpyDeviceToHost);
        print_and_test("Kamping raw pointer", h_result);
    }
    cudaFree(d_buf);
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    mpi_test_raw_pointer(rank);
    MPI_Barrier(MPI_COMM_WORLD);

    kamping_test_span(rank);
    MPI_Barrier(MPI_COMM_WORLD);

    kamping_test_thrust_device(rank);
    MPI_Barrier(MPI_COMM_WORLD);

    mpi_test_thrust_device(rank);
    MPI_Barrier(MPI_COMM_WORLD);
    
    kamping_test_raw_pointer(rank);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;   
}

