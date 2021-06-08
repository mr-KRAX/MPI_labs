#define MSMPI_NO_DEPRECATE_20

#include <complex>
#include <vector>
#include <cmath>

#include <stdio.h>;
#include "mpi.h";
#include "SchonhageStrassen.h"

using namespace std;

int main(int argc, char* argv[]) {
  srand(time(NULL));

  const int N = 4; // длина чисел

  lNum a, b, c;
  int* array = new int[N];
  int* result1 = new int[N];
  int* result2 = new int[N];
  int size1, size2;

  int ProcNum, ProcRank;
  MPI_Status Status;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

  // двумерная декартовая решётка. (2х2)
  MPI_Comm GridComm;
  const int ndims = 2;
  int dims[ndims], periodic[ndims], reorder = 1, maxdims = 2;
  dims[0] = dims[1] = 2;
  periodic[0] = periodic[1] = 1;
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periodic, reorder, &GridComm);
  int coords[ndims * ndims][2];

  if (ProcRank == 0) { // (0,0).
    printf("\nTopology: \n");
    for (int rank = 0; rank < ndims * ndims; rank++) {
      MPI_Cart_coords(GridComm, rank, ndims, coords[rank]);
      printf("Node 0: node %d is [%d, %d]\n", rank, coords[rank][0], coords[rank][1]);
    }

    // генерируем число
    printf("\nNode 0 - generated number: ");
    for (int i = N-1; i >= 0; i--) {
      array[i] = rand() % 9;
      printf("%d", array[i]);
    }
    printf("\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(array, N, MPI_INT, 0, MPI_COMM_WORLD);

  if (ProcRank == 1 || ProcRank == 2) { // (0,1) и (1,0).
    size1 = N;

    a = SS_intArray2LNum(array, size1);
    b = SS_intArray2LNum(array, size1);
    c = SS_fastMultiply(a, b);
    printf("\nNode %d - multiplication: %s\n", ProcRank, SS_lNum2String(c).c_str());

    if (ProcRank == 1) { // (0,1).
      result1 = SS_lNum2Array(c, &size1);
      MPI_Send(&size1, 1, MPI_INT, 3, 0, MPI_COMM_WORLD);
      MPI_Send(result1, size1, MPI_INT, 3, 0, MPI_COMM_WORLD);
    }

    if (ProcRank == 2) { // (1,0).
      result2 = SS_lNum2Array(c, &size2);
      MPI_Send(&size2, 1, MPI_INT, 3, 0, MPI_COMM_WORLD);
      MPI_Send(result2, size2, MPI_INT, 3, 0, MPI_COMM_WORLD);
    }
  }

  if (ProcRank == 3) { // (1,1)
    printf("\nNode %d - result of multiplication", ProcRank);

    // принимаем с (0,1)
    MPI_Recv(&size1, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &Status);
    result1 = new int[size1];
    MPI_Recv(result1, size1, MPI_INT, 1, 0, MPI_COMM_WORLD, &Status);

    printf("\nFrom Node 1: ");
    for (int i = size1 - 1; i >= 0; i--)
      printf("%d", result1[i]);


    // принимаем с (1,0)
    MPI_Recv(&size2, 1, MPI_INT, 2, 0, MPI_COMM_WORLD, &Status);
    result2 = new int[size2];
    MPI_Recv(result2, size2, MPI_INT, 2, 0, MPI_COMM_WORLD, &Status);

    printf("\nFrom Node 2: ");
    for (int i = size2 - 1; i >= 0; i--)
      printf("%d", result2[i]);

    a = SS_intArray2LNum(result1, size1);
    b = SS_intArray2LNum(result2, size2);
    c = SS_fastMultiply(a, b);
    printf("\nTotal res: %s\n", SS_lNum2String(c).c_str());

    MPI_Comm_free(&GridComm);
  }

  delete[] array;
  delete[] result1;
  delete[] result2;
  MPI_Finalize();
  return 0;
}