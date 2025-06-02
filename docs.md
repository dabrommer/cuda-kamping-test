# CUDA-aware KaMPIng

## Supported Send/Recv Buffers 
| Data Type   | Supported as send/recv buffer? | Notes                |
|-------------|:------------------:|----------------------|
| `std::span`                                          | ✅                 |                      |
| `std::span from device memory to host memory`        | ✅                 |                      |
| `std::span from device memory to host std::vector`   | ✅                 |                      |
| `thrust::host_vector`                                | ✅                 |                      |
| `raw gpu pointer`                                    | ❌                 | static assertion failed: MPI does not support pointer types. Why do you want to transfer a pointer over MPI?|     
| `raw gpu pointer with size and type`                 | ❌                 | MPI is called but result is wrong: DataBuffer::value_type_with_const* data() calls return &underlying() which won't return the original gpu pointer -> MPI is called with wrong pointer|                           
| `thrust::device_vector`                              | ❌                 | Doesn't work with MPI either, because device_vector.data() returns a thrust::device_ptr<float>. Workaround: use device_vector.data().get() to get raw pointer. This won't work with kamping because of the issue with raw gpu pointers.|



