// Copyright 2023 kulagin_a
#include "task_2/kulagin_a_my_reduce/my_reduce.h"
#include <algorithm>
#include <iostream>
#include <cstdlib>

static const char* incorrect_op_msg = "Incorrect MPI_Op op\n";
static const char* incorrect_type_msg = "Incorrect MPI_Datatype type\n";
static const int reduce_tag = 0;

// I hate cppcheck
typedef long double ldouble;

template<typename T>
static inline int my_mpi_op(T* a, T* b, int count, MPI_Op op, bool print_err = true) {
  #ifndef shortened_for_loop
  #define shortened_for_loop(i, n) for ((i) = 0; (i) < (n); (i)++)
  #endif
  int i, ret_status = MPI_SUCCESS;
  if (op == MPI_MAX) {
    shortened_for_loop(i, count)
      b[i] = std::max<T>(a[i], b[i]);
  } else if (op == MPI_MIN) {
    shortened_for_loop(i, count)
      b[i] = std::min<T>(a[i], b[i]);
  } else if (op == MPI_SUM) {
    shortened_for_loop(i, count)
      b[i] = a[i] + b[i];
  } else if (op == MPI_PROD) {
    shortened_for_loop(i, count)
      b[i] = a[i] * b[i];
  } else if (op == MPI_LAND) {
    shortened_for_loop(i, count)
      b[i] = a[i] && b[i];
  } else if (op == MPI_LOR) {
    shortened_for_loop(i, count)
      b[i] = a[i] || b[i];
  } else if (op == MPI_LXOR) {
    shortened_for_loop(i, count)
      b[i] = !a[i] != !b[i];
  } else {
    if (print_err) {
      std::cout << incorrect_op_msg;
    }
    ret_status = MPI_ERR_OP;
  }
  #undef shortened_for_loop
  return ret_status;
}

// Because BAND, BOR and BXOR are not defined in case of floats and doubles.
template<typename T>
static inline int my_mpi_op_int(T* a, T* b, int count, MPI_Op op) {
  int ret_status = my_mpi_op<T>(a, b, count, op, false);
  if (ret_status != MPI_SUCCESS) {
    int i;
    ret_status = MPI_SUCCESS;
    #ifndef shortened_for_loop
    #define shortened_for_loop(i, n) for ((i) = 0; (i) < (n); (i)++)
    #endif
    if (op == MPI_BAND) {
      shortened_for_loop(i, count)
        b[i] = a[i] & b[i];
    } else if (op == MPI_BOR) {
      shortened_for_loop(i, count)
        b[i] = a[i] | b[i];
    } else if (op == MPI_BXOR) {
      shortened_for_loop(i, count)
        b[i] = a[i] ^ b[i];
    } else {
      std::cout << incorrect_op_msg;
      ret_status = MPI_ERR_OP;
    }
    #undef shortened_for_loop
  }
  return ret_status;
}

int my_mpi_reduce(const void* send, void* recv, int count, MPI_Datatype type, MPI_Op op, int root, MPI_Comm comm) {
  int proc_rank, proc_num, type_sizeof, ret, flag;
  MPI_Comm_size(comm, &proc_num);
  MPI_Comm_rank(comm, &proc_rank);
  MPI_Type_size(type, &type_sizeof);
  if (root < 0 || root >= proc_num) {
    ret = MPI_ERR_ROOT;
    MPI_Abort(comm, ret);
    return ret;
  }
  if (count <= 0) {
    ret = MPI_ERR_COUNT;
    MPI_Abort(comm, ret);
    return ret;
  }
  if (send == nullptr || recv == nullptr) {
    ret = MPI_ERR_BUFFER;
    MPI_Abort(comm, ret);
    return ret;
  }
  if (root == proc_rank) {
    int i;
    void* tmp = std::malloc(count*type_sizeof);
    memcpy(recv, send, count*type_sizeof);
    for (i = 0; i < proc_num; i++) {
      if (i != root) {
        ret = MPI_Recv(tmp, count, type, i, reduce_tag, comm, MPI_STATUS_IGNORE);
        if (ret != MPI_SUCCESS) {
          std::free(tmp);
          MPI_Abort(comm, ret);
          return ret;
        }
        if (type == MPI_CHAR) {
          ret = my_mpi_op_int<char>(reinterpret_cast<char*>(tmp), reinterpret_cast<char*>(recv), count, op);
        } else if (type == MPI_SHORT) {
          ret = my_mpi_op_int<int16_t>(reinterpret_cast<int16_t*>(tmp), reinterpret_cast<int16_t*>(recv), count, op);
        } else if (type == MPI_LONG) {
          ret = my_mpi_op_int<int32_t>(reinterpret_cast<int32_t*>(tmp), reinterpret_cast<int32_t*>(recv), count, op);
        } else if (type == MPI_INT) {
          ret = my_mpi_op_int<int>(reinterpret_cast<int*>(tmp), reinterpret_cast<int*>(recv), count, op);
        } else if (type == MPI_UNSIGNED_CHAR) {
          ret = my_mpi_op_int<uint8_t>(reinterpret_cast<uint8_t*>(tmp), reinterpret_cast<uint8_t*>(recv), count, op);
        } else if (type == MPI_UNSIGNED_SHORT) {
          ret = my_mpi_op_int<uint16_t>(reinterpret_cast<uint16_t*>(tmp), reinterpret_cast<uint16_t*>(recv), count, op);
        } else if (type == MPI_UNSIGNED) {
          ret = my_mpi_op_int<unsigned>(reinterpret_cast<unsigned*>(tmp), reinterpret_cast<unsigned*>(recv), count, op);
        } else if (type == MPI_UNSIGNED_LONG) {
          ret = my_mpi_op_int<uint32_t>(reinterpret_cast<uint32_t*>(tmp), reinterpret_cast<uint32_t*>(recv), count, op);
        } else if (type == MPI_FLOAT) {
          ret = my_mpi_op<float>(reinterpret_cast<float*>(tmp), reinterpret_cast<float*>(recv), count, op);
        } else if (type == MPI_DOUBLE) {
          ret = my_mpi_op<double>(reinterpret_cast<double*>(tmp), reinterpret_cast<double*>(recv), count, op);
        } else if (type == MPI_LONG_DOUBLE) {
          ret = my_mpi_op<ldouble>(reinterpret_cast<ldouble*>(tmp), reinterpret_cast<ldouble*>(recv), count, op);
        } else {
          std::cout << incorrect_type_msg;
          ret = MPI_ERR_TYPE;
        }
        if (ret != MPI_SUCCESS) {
          std::free(tmp);
          MPI_Abort(comm, ret);
          return ret;
        }
      }
    }
    std::free(tmp);
  } else {
    MPI_Send(send, count, type, root, reduce_tag, comm);
  }
  // MPI_Barrier(comm);
  return MPI_SUCCESS;
}
