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

#define RUN_TEST(test) \
    test(rank);        \
    MPI_Barrier(MPI_COMM_WORLD);


void print_and_test(const char* test_name, const float* h_result) {
    std::cout << test_name << " Rank 1 received: ";
    for (int i = 0; i < N; ++i) std::cout << h_result[i] << " ";
    std::cout << std::endl;

    for (int i = 0; i < N; ++i) {
        KASSERT(h_result[i] == X, "Error in " << test_name);
    }
}

// MPI with raw gpu pointers
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

// KaMPIng with gpu pointers in cuda::std::span
void kamping_test_span(int rank) {
    kamping::Communicator comm;
    float* d_buf;
    cudaMalloc((void**)&d_buf, N * sizeof(float));
    cuda::std::span<float> d_span(d_buf, N);

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

// KaMPIng with thrust host vector
void kamping_test_thrust_host_vec(int rank) {
    kamping::Communicator comm;

    if (rank == 0) {
        thrust::host_vector<float> h_vec(N, X);
        comm.send(kamping::send_buf(h_vec), kamping::destination(1));

    } else if (rank == 1) {
        thrust::host_vector<float> h_vec(N, 99);
        comm.recv(kamping::recv_buf<kamping::no_resize>(h_vec));
        print_and_test("Kamping thrust host vec", h_vec.data());
    }
}
 
// MPI with thrust host vector
void mpi_test_thrust_host_vec(int rank) {
    if (rank == 0) {
        thrust::host_vector<float> h_vec(N, X);
        MPI_Send(h_vec.data(), N, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);

    } else if (rank == 1) {
        thrust::host_vector<float> h_vec(N, 99);
        MPI_Recv(h_vec.data(), N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        print_and_test("MPI thrust host vec", h_vec.data());
    }
}

// KaMPIng with raw gpu pointers with send/recv count and type
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

// MPI with raw gpu pointer to raw host pointer
void mpi_test_raw_pointer_to_host(int rank) {
    if (rank == 0) {
        float* d_buf;
        cudaMalloc((void**)&d_buf, N * sizeof(float));
        float h_data[N];
        for (int i = 0; i < N; ++i) h_data[i] = X;
        cudaMemcpy(d_buf, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
		MPI_Send(d_buf, N, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
        cudaFree(d_buf);

    } else if (rank == 1) {
        float* h_buf = (float*) malloc(N * sizeof(float));
        for (int i = 0; i < N; ++i) h_buf[i] = 99; 
        MPI_Recv(h_buf, N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        print_and_test("MPI raw gpu pointer to host", h_buf);
        free(h_buf);
    }   
}

// KaMPIng with gpu span to host span
void kamping_test_span_to_host(int rank) {
    kamping::Communicator comm;

    if (rank == 0) {
        float* d_buf;
        cudaMalloc((void**)&d_buf, N * sizeof(float));
        cuda::std::span<float> d_span(d_buf, N);
        float h_data[N];
        for (int i = 0; i < N; ++i) h_data[i] = X;
        cudaMemcpy(d_buf, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
        comm.send(kamping::send_buf(d_span), kamping::destination(1));
        cudaFree(d_buf);

    } else if (rank == 1) {
        float* h_buf = (float*) malloc(N * sizeof(float));
        for (int i = 0; i < N; ++i) h_buf[i] = 99;
        cuda::std::span<float> d_span(h_buf, N);
        comm.recv(kamping::recv_buf<kamping::no_resize>(d_span));
        print_and_test("Kamping gpu span to host", d_span.data());
        free(h_buf);
    }    
}

// KaMPIng with gpu span to host vector
void kamping_test_span_to_host_vec(int rank) {
    kamping::Communicator comm;

    if (rank == 0) {
        float* d_buf;
        cudaMalloc((void**)&d_buf, N * sizeof(float));
        cuda::std::span<float> d_span(d_buf, N);
        float h_data[N];
        for (int i = 0; i < N; ++i) h_data[i] = X;
        cudaMemcpy(d_buf, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
        comm.send(kamping::send_buf(d_span), kamping::destination(1));
        cudaFree(d_buf);

    } else if (rank == 1) {
        std::vector<float> h_buf(N, 99);
        comm.recv(kamping::recv_buf<kamping::no_resize>(h_buf));
        print_and_test("Kamping gpu span to host vector", h_buf.data());
    }    
}

// MPI with thrust device vector
void mpi_test_device_vector(int rank) {
    if (rank == 0) {
        thrust::device_vector<float> d_vec(N, X);
        MPI_Send(d_vec.data().get(), N, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);

    } else if (rank == 1) {
        thrust::device_vector<float> d_vec(N, 99);
        MPI_Recv(d_vec.data().get(), N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        float h_result[N];
        cudaMemcpy(h_result, d_vec.data().get(), N * sizeof(float), cudaMemcpyDeviceToHost);
        print_and_test("MPI thrust device vector", h_result);
    }
}

// KaMPIng with thrust device vector
void kamping_test_device_vector(int rank) {
    kamping::Communicator comm;

    if (rank == 0) {
        thrust::device_vector<float> d_vec(N, X);
        comm.send(kamping::send_buf(d_vec.data().get()), kamping::destination(1), kamping::send_count(N), kamping::send_type(MPI_FLOAT));

    } else if (rank == 1) {
        thrust::device_vector<float> d_vec(N, 99);
        comm.recv(kamping::recv_buf<kamping::no_resize>(d_vec.data().get()), kamping::recv_count(N), kamping::recv_type(MPI_FLOAT));
        float h_result[N];
        cudaMemcpy(h_result, d_vec.data().get(), N * sizeof(float), cudaMemcpyDeviceToHost);
        print_and_test("Kamping thrust device vector", h_result);
    }
}


int main(int argc, char **argv) { 
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    RUN_TEST(mpi_test_raw_pointer);
    RUN_TEST(kamping_test_span);
    RUN_TEST(kamping_test_thrust_host_vec);
    RUN_TEST(mpi_test_thrust_host_vec)
    // RUN_TEST(kamping_test_raw_pointer);
    RUN_TEST(mpi_test_raw_pointer_to_host);
    RUN_TEST(kamping_test_span_to_host)
    RUN_TEST(kamping_test_span_to_host_vec)
    RUN_TEST(mpi_test_device_vector);
    // RUN_TEST(kamping_test_device_vector);

    MPI_Finalize();
    return 0;   
}

