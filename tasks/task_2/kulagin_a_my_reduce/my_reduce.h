// Copyright 2023 kulagin_a
#ifndef TASKS_TASK_2_KULAGIN_A_MY_REDUCE_MY_REDUCE_H_
#define TASKS_TASK_2_KULAGIN_A_MY_REDUCE_MY_REDUCE_H_

#include <mpi.h>

int my_mpi_reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype type, MPI_Op op, int root, MPI_Comm comm);

#endif  // TASKS_TASK_2_KULAGIN_A_MY_REDUCE_MY_REDUCE_H_
