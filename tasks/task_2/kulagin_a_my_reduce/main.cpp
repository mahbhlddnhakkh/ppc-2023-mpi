// Copyright 2023 kulagin_a
#include <gtest/gtest.h>
#include <mpi.h>
#include <cstdlib>
#include "./my_reduce.h"

typedef int vector_type;
static const MPI_Datatype mpi_vector_type = MPI_INT;
static const int vector_size_mul_max = 50001;
static const vector_type vector_element_max = 100;
static const vector_type vector_element_min = -100;

inline void test_vector_scalar() {
  double res_time[2];
  vector_type *a, *b;
  int i, n, proc_num, proc_rank;
  vector_type sum, sum_all, sum_real;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  if (proc_rank == 0) {
    n = proc_num * (std::rand() % vector_size_mul_max + 1);
    a = new vector_type[n];
    b = new vector_type[n];
    sum_real = 0;
    for (i = 0; i < n; i++) {
      a[i] = std::rand() % (vector_element_max - vector_element_min + 1) + vector_element_min;
      b[i] = std::rand() % (vector_element_max - vector_element_min + 1) + vector_element_min;
      sum_real += a[i] * b[i];
    }
  }
  MPI_Bcast(&n, 1, mpi_vector_type, 0, MPI_COMM_WORLD);
  n = n / proc_num;
  if (proc_rank != 0) {
    a = new vector_type[n];
    b = new vector_type[n];
  }
  MPI_Scatter(a, n, mpi_vector_type, a, n, mpi_vector_type, 0, MPI_COMM_WORLD);
  MPI_Scatter(b, n, mpi_vector_type, b, n, mpi_vector_type, 0, MPI_COMM_WORLD);
  sum = sum_all = 0;
  for (i = 0; i < n; i++) {
    sum += a[i] * b[i];
  }

  if (proc_rank == 0) {
    res_time[0] = MPI_Wtime();
  }
  MPI_Reduce(&sum, &sum_all, 1, mpi_vector_type, MPI_SUM, 0, MPI_COMM_WORLD);
  if (proc_rank == 0) {
    res_time[1] = MPI_Wtime();
    std::cout << "MPI_Reduce time = " << (res_time[1] - res_time[0]) << '\n';
  }
  if (proc_rank == 0) {
    res_time[0] = MPI_Wtime();
  }
  EXPECT_EQ(MPI_SUCCESS, my_mpi_reduce(&sum, &sum_all, 1, mpi_vector_type, MPI_SUM, 0, MPI_COMM_WORLD));
  if (proc_rank == 0) {
    res_time[1] = MPI_Wtime();
    std::cout << "my_mpi_reduce time = " << (res_time[1] - res_time[0]) << '\n';
  }
  if (proc_rank == 0) {
    EXPECT_EQ(sum_real, sum_all);
  }
  delete[] a;
  delete[] b;
}

TEST(Parallel_Operation_Reduce_MPI, test1_vector_scalar) {
  test_vector_scalar();
}

TEST(Parallel_Operation_Reduce_MPI, test2_vector_scalar) {
  test_vector_scalar();
}

TEST(Parallel_Operation_Reduce_MPI, test3_vector_scalar) {
  test_vector_scalar();
}

TEST(Parallel_Operation_Reduce_MPI, test4_vector_scalar) {
  test_vector_scalar();
}

TEST(Parallel_Operation_Reduce_MPI, test5_vector_scalar) {
  test_vector_scalar();
}

int main(int argc, char** argv) {
  int world_rank, result;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (world_rank != 0) {
    delete listeners.Release(listeners.default_result_printer());
  } else {
    std::srand(std::time(nullptr));
  }
  result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
