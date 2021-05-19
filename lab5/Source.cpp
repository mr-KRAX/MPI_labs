#include "mpi.h"
#include <algorithm>
#include <complex>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <string>

// #define cplx cplx

using namespace std;

typedef complex<double> cplx;

#define freeMtx(m) \
  do {             \
    free(m[0]);    \
    free(m);       \
  } while (0)

int GetNewDim(int oldDim) {
  double integerPart;
  double log = log2(oldDim);
  modf(log, &integerPart);
  if (log - integerPart != 0)
    return (int)pow(2, ceil(log));
  else
    return (int)oldDim;
}

void SplitMatrix(cplx **matrix, cplx **M11, cplx **M12, cplx **M21, cplx **M22,
                 cplx *data11, cplx *data12, cplx *data21, cplx *data22, int matrixSize) {
  int partSize = matrixSize >> 1;

  for (int i = 0; i < matrixSize; i++) {
    if (i < partSize) {
      M11[i] = &(data11[partSize * i]);
      M12[i] = &(data12[partSize * i]);
      M21[i] = &(data21[partSize * i]);
      M22[i] = &(data22[partSize * i]);
    }
    for (int j = 0; j < matrixSize; j++) {
      if (i < partSize && j < partSize)
        M11[i][j] = matrix[i][j];
      else if (i < partSize && j >= partSize)
        M12[i][j - partSize] = matrix[i][j];
      else if (i >= partSize && j < partSize)
        M21[i - partSize][j] = matrix[i][j];
      else if (i >= partSize && j >= partSize)
        M22[i - partSize][j - partSize] = matrix[i][j];
    }
  }
}

void RestoreMatrix(cplx **matrix, cplx **M11, cplx **M12, cplx **M21, cplx **M22, int partSize) {
  int newMatrixSize = partSize << 1;

  for (int i = 0; i < newMatrixSize; i++)
    for (int j = 0; j < newMatrixSize; j++) {
      if (i < partSize && j < partSize)
        matrix[i][j] = M11[i][j];
      else if (i < partSize && j >= partSize)
        matrix[i][j] = M12[i][j - partSize];
      else if (i >= partSize && j < partSize)
        matrix[i][j] = M21[i - partSize][j];
      else if (i >= partSize && j >= partSize)
        matrix[i][j] = M22[i - partSize][j - partSize];
    }
}

cplx **Sum(cplx **matrixA, cplx **matrixB, int matrixSize) {
  cplx *dataC = (cplx *)malloc(matrixSize * matrixSize * sizeof(cplx));
  cplx **matrixC = (cplx **)malloc(sizeof(cplx *) * matrixSize);
  for (int i = 0; i < matrixSize; i++) {
    matrixC[i] = &(dataC[matrixSize * i]);
    for (int j = 0; j < matrixSize; j++)
      matrixC[i][j] = matrixA[i][j] + matrixB[i][j];
  }
  return matrixC;
}

cplx **Subtract(cplx **matrixA, cplx **matrixB, int matrixSize) {
  cplx *dataC = (cplx *)malloc(matrixSize * matrixSize * sizeof(cplx));
  cplx **matrixC = (cplx **)malloc(sizeof(cplx *) * matrixSize);
  for (int i = 0; i < matrixSize; i++) {
    matrixC[i] = &(dataC[matrixSize * i]);
    for (int j = 0; j < matrixSize; j++)
      matrixC[i][j] = matrixA[i][j] - matrixB[i][j];
  }
  return matrixC;
}

cplx **MultiplyRegularly(cplx **matrixA, cplx **matrixB, int matrixSize) {
  cplx *dataC = (cplx *)malloc(matrixSize * matrixSize * sizeof(cplx));
  cplx **matrixC = (cplx **)malloc(sizeof(cplx *) * matrixSize);
  bool isElementInitialized = false;

  for (int i = 0; i < matrixSize; i++) {
    matrixC[i] = &(dataC[matrixSize * i]);
    for (int j = 0; j < matrixSize; j++) {
      isElementInitialized = false;
      for (int k = 0; k < matrixSize; k++) {
        if (isElementInitialized) {
          matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
        } else {
          matrixC[i][j] = matrixA[i][k] * matrixB[k][j];
          isElementInitialized = true;
        }
      }
    }
  }
  return matrixC;
}

cplx **Multiply(cplx **matrixA, cplx **matrixB, int matrixSize,
                int NewProcRank, int NewProcNum, bool isParalleled, MPI_Comm new_comm) {
  cplx *dataC = (cplx *)malloc(matrixSize * matrixSize * sizeof(cplx));
  cplx **matrixC = (cplx **)malloc(sizeof(cplx *) * matrixSize);
  for (int i = 0; i < matrixSize; i++)
    matrixC[i] = &(dataC[matrixSize * i]);

  if (matrixSize > 40) {
    int partSize = matrixSize >> 1;

    cplx *dataA11 = (cplx *)malloc(partSize * partSize * sizeof(cplx));
    cplx **A11 = (cplx **)malloc(sizeof(cplx *) * partSize);
    cplx *dataA12 = (cplx *)malloc(partSize * partSize * sizeof(cplx));
    cplx **A12 = (cplx **)malloc(sizeof(cplx *) * partSize);
    cplx *dataA21 = (cplx *)malloc(partSize * partSize * sizeof(cplx));
    cplx **A21 = (cplx **)malloc(sizeof(cplx *) * partSize);
    cplx *dataA22 = (cplx *)malloc(partSize * partSize * sizeof(cplx));
    cplx **A22 = (cplx **)malloc(sizeof(cplx *) * partSize);
    SplitMatrix(matrixA, A11, A12, A21, A22, dataA11, dataA12, dataA21, dataA22, matrixSize);

    cplx *dataB11 = (cplx *)malloc(partSize * partSize * sizeof(cplx));
    cplx **B11 = (cplx **)malloc(sizeof(cplx *) * partSize);
    cplx *dataB12 = (cplx *)malloc(partSize * partSize * sizeof(cplx));
    cplx **B12 = (cplx **)malloc(sizeof(cplx *) * partSize);
    cplx *dataB21 = (cplx *)malloc(partSize * partSize * sizeof(cplx));
    cplx **B21 = (cplx **)malloc(sizeof(cplx *) * partSize);
    cplx *dataB22 = (cplx *)malloc(partSize * partSize * sizeof(cplx));
    cplx **B22 = (cplx **)malloc(sizeof(cplx *) * partSize);
    SplitMatrix(matrixB, B11, B12, B21, B22, dataB11, dataB12, dataB21, dataB22, matrixSize);

    cplx *dataC11 = (cplx *)malloc(partSize * partSize * sizeof(cplx));
    cplx **C11 = (cplx **)malloc(sizeof(cplx *) * partSize);
    cplx *dataC12 = (cplx *)malloc(partSize * partSize * sizeof(cplx));
    cplx **C12 = (cplx **)malloc(sizeof(cplx *) * partSize);
    cplx *dataC21 = (cplx *)malloc(partSize * partSize * sizeof(cplx));
    cplx **C21 = (cplx **)malloc(sizeof(cplx *) * partSize);
    cplx *dataC22 = (cplx *)malloc(partSize * partSize * sizeof(cplx));
    cplx **C22 = (cplx **)malloc(sizeof(cplx *) * partSize);
    SplitMatrix(matrixC, C11, C12, C21, C22, dataC11, dataC12, dataC21, dataC22, matrixSize);

    if (!isParalleled) {
      switch (NewProcRank) {
      case 0: {
        cplx **P1, **P2, **P3, **P4, **P5, **P6, **P7;
        P1 = Multiply(Sum(A11, A22, partSize), Sum(B11, B22, partSize), partSize, NewProcRank, NewProcNum, true, new_comm);

        cplx *dataP2 = (cplx *)malloc(partSize * partSize * sizeof(cplx));
        P2 = (cplx **)malloc(sizeof(cplx *) * partSize);
        cplx *dataP3 = (cplx *)malloc(partSize * partSize * sizeof(cplx));
        P3 = (cplx **)malloc(sizeof(cplx *) * partSize);
        cplx *dataP4 = (cplx *)malloc(partSize * partSize * sizeof(cplx));
        P4 = (cplx **)malloc(sizeof(cplx *) * partSize);
        cplx *dataP5 = (cplx *)malloc(partSize * partSize * sizeof(cplx));
        P5 = (cplx **)malloc(sizeof(cplx *) * partSize);
        cplx *dataP6 = (cplx *)malloc(partSize * partSize * sizeof(cplx));
        P6 = (cplx **)malloc(sizeof(cplx *) * partSize);
        cplx *dataP7 = (cplx *)malloc(partSize * partSize * sizeof(cplx));
        P7 = (cplx **)malloc(sizeof(cplx *) * partSize);

        for (int i = 0; i < partSize; i++) {
          P2[i] = &(dataP2[i * partSize]);
          P3[i] = &(dataP3[i * partSize]);
          P4[i] = &(dataP4[i * partSize]);
          P5[i] = &(dataP5[i * partSize]);
          P6[i] = &(dataP6[i * partSize]);
          P7[i] = &(dataP7[i * partSize]);
        }
        MPI_Barrier(new_comm);
        MPI_Recv(&P2[0][0], partSize * partSize, MPI_DOUBLE_COMPLEX, 1, MPI_ANY_TAG, new_comm, MPI_STATUS_IGNORE);
        MPI_Recv(&P3[0][0], partSize * partSize, MPI_DOUBLE_COMPLEX, 2, MPI_ANY_TAG, new_comm, MPI_STATUS_IGNORE);
        MPI_Recv(&P4[0][0], partSize * partSize, MPI_DOUBLE_COMPLEX, 3, MPI_ANY_TAG, new_comm, MPI_STATUS_IGNORE);
        MPI_Recv(&P5[0][0], partSize * partSize, MPI_DOUBLE_COMPLEX, 4, MPI_ANY_TAG, new_comm, MPI_STATUS_IGNORE);
        MPI_Recv(&P6[0][0], partSize * partSize, MPI_DOUBLE_COMPLEX, 5, MPI_ANY_TAG, new_comm, MPI_STATUS_IGNORE);
        MPI_Recv(&P7[0][0], partSize * partSize, MPI_DOUBLE_COMPLEX, 6, MPI_ANY_TAG, new_comm, MPI_STATUS_IGNORE);

        C11 = Sum(Subtract(Sum(P1, P4, partSize), P5, partSize), P7, partSize);
        C12 = Sum(P3, P5, partSize);
        C21 = Sum(P2, P4, partSize);
        C22 = Sum(Sum(Subtract(P1, P2, partSize), P3, partSize), P6, partSize);

        RestoreMatrix(matrixC, C11, C12, C21, C22, partSize);

        freeMtx(P1);
        freeMtx(P2);
        freeMtx(P3);
        freeMtx(P4);
        freeMtx(P5);
        freeMtx(P6);
        freeMtx(P7);

        freeMtx(C11);
        freeMtx(C12);
        freeMtx(C21);
        freeMtx(C22);

        break;
      }
      case 1: {
        cplx **P2;
        P2 = Multiply(Sum(A21, A22, partSize), B11, partSize, NewProcRank, NewProcNum, true, new_comm);
        MPI_Barrier(new_comm);
        MPI_Send(&P2[0][0], partSize * partSize, MPI_DOUBLE_COMPLEX, 0, 0, new_comm);
      } break;
      case 2: {
        cplx **P3;
        P3 = Multiply(A11, Subtract(B12, B22, partSize), partSize, NewProcRank, NewProcNum, true, new_comm);
        MPI_Barrier(new_comm);
        MPI_Send(&P3[0][0], partSize * partSize, MPI_DOUBLE_COMPLEX, 0, 0, new_comm);
      } break;
      case 3: {
        cplx **P4;
        P4 = Multiply(A22, Subtract(B21, B11, partSize), partSize, NewProcRank, NewProcNum, true, new_comm);
        MPI_Barrier(new_comm);
        MPI_Send(&P4[0][0], partSize * partSize, MPI_DOUBLE_COMPLEX, 0, 0, new_comm);
      } break;
      case 4: {
        cplx **P5;
        P5 = Multiply(Sum(A11, A12, partSize), B22, partSize, NewProcRank, NewProcNum, true, new_comm);
        MPI_Barrier(new_comm);
        MPI_Send(&P5[0][0], partSize * partSize, MPI_DOUBLE_COMPLEX, 0, 0, new_comm);
      } break;
      case 5: {
        cplx **P6;
        P6 = Multiply(Subtract(A21, A11, partSize), Sum(B11, B12, partSize), partSize, NewProcRank, NewProcNum, true, new_comm);
        MPI_Barrier(new_comm);
        MPI_Send(&P6[0][0], partSize * partSize, MPI_DOUBLE_COMPLEX, 0, 0, new_comm);
      } break;
      case 6: {
        cplx **P7;
        P7 = Multiply(Subtract(A12, A22, partSize), Sum(B21, B22, partSize), partSize, NewProcRank, NewProcNum, true, new_comm);
        MPI_Barrier(new_comm);
        MPI_Send(&P7[0][0], partSize * partSize, MPI_DOUBLE_COMPLEX, 0, 0, new_comm);
      } break;
      }
    } else {
      cplx **P1, **P2, **P3, **P4, **P5, **P6, **P7;
      P1 = Multiply(Sum(A11, A22, partSize), Sum(B11, B22, partSize), partSize, NewProcRank, NewProcNum, true, new_comm);
      P2 = Multiply(Sum(A21, A22, partSize), B11, partSize, NewProcRank, NewProcNum, true, new_comm);
      P3 = Multiply(A11, Subtract(B12, B22, partSize), partSize, NewProcRank, NewProcNum, true, new_comm);
      P4 = Multiply(A22, Subtract(B21, B11, partSize), partSize, NewProcRank, NewProcNum, true, new_comm);
      P5 = Multiply(Sum(A11, A12, partSize), B22, partSize, NewProcRank, NewProcNum, true, new_comm);
      P6 = Multiply(Subtract(A21, A11, partSize), Sum(B11, B12, partSize), partSize, NewProcRank, NewProcNum, true, new_comm);
      P7 = Multiply(Subtract(A12, A22, partSize), Sum(B21, B22, partSize), partSize, NewProcRank, NewProcNum, true, new_comm);

      C11 = Sum(Subtract(Sum(P1, P4, partSize), P5, partSize), P7, partSize);
      C12 = Sum(P3, P5, partSize);
      C21 = Sum(P2, P4, partSize);
      C22 = Sum(Sum(Subtract(P1, P2, partSize), P3, partSize), P6, partSize);

      RestoreMatrix(matrixC, C11, C12, C21, C22, partSize);

      freeMtx(P1);
      freeMtx(P2);
      freeMtx(P3);
      freeMtx(P4);
      freeMtx(P5);
      freeMtx(P6);
      freeMtx(P7);

      freeMtx(C11);
      freeMtx(C12);
      freeMtx(C21);
      freeMtx(C22);
    }
    matrixC = MultiplyRegularly(matrixA, matrixB, matrixSize);
  } else
    matrixC = MultiplyRegularly(matrixA, matrixB, matrixSize);

  return matrixC;
}

cplx **CropMatrixBack(cplx **matrix, int oldMatrixSize) {
  cplx *data = (cplx *)malloc(oldMatrixSize * oldMatrixSize * sizeof(cplx));
  cplx **newMatrix = (cplx **)malloc(sizeof(cplx *) * oldMatrixSize);
  for (int i = 0; i < oldMatrixSize; i++) {
    newMatrix[i] = &(data[oldMatrixSize * i]);
    for (int j = 0; j < oldMatrixSize; j++)
      newMatrix[i][j] = matrix[i][j];
  }
  return newMatrix;
}

bool Compare(cplx **matrixA, cplx **matrixB, int matrixSize) {
  for (int i = 0; i < matrixSize; i++)
    for (int j = 0; j < matrixSize; j++)
      if (matrixA[i][j] != matrixB[i][j])
        return false;

  return true;
}

string MatrixToString(cplx **matrix, int matrixSize) {
  string output = "";
  for (int i = 0; i < matrixSize; i++) {
    for (int j = 0; j < matrixSize; j++)
      output += to_string(real(matrix[i][j])) + " + " + to_string(imag(matrix[i][j])) + "i\t";
    output += "\r\n";
  }

  return output;
}

// Usage: mpiexec.exe -n 8 Lab4.exe m
int main(int argc, char *argv[]) {
  int ProcNum, ProcRank;
  int NewProcNum, NewProcRank;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  if (ProcNum == 8) {
    MPI_Group new_group;
    MPI_Group original_group;

    int ranks_for_new_group[7] = {1, 2, 3, 4, 5, 6, 7};

    MPI_Comm_group(MPI_COMM_WORLD, &original_group);

    /// Creates a new group.
    MPI_Group_incl(original_group, 7, ranks_for_new_group, &new_group);

    /// Creates new communicator.
    MPI_Comm new_comm;
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);

    if (new_comm != MPI_COMM_NULL) {
      MPI_Comm_size(new_comm, &NewProcNum);
      MPI_Comm_rank(new_comm, &NewProcRank);
    }

    // Init matrixes
    int oldN = atoi(argv[1]);
    int N = GetNewDim(oldN);

    cplx *dataA = (cplx *)malloc(N * N * sizeof(cplx));
    cplx **A = (cplx **)malloc(sizeof(cplx *) * N);
    cplx *dataB = (cplx *)malloc(N * N * sizeof(cplx));
    cplx **B = (cplx **)malloc(sizeof(cplx *) * N);

    cplx *dataConventionalA = (cplx *)malloc(oldN * oldN * sizeof(cplx));
    cplx **regA = (cplx **)malloc(sizeof(cplx *) * oldN);
    cplx *dataConventionalB = (cplx *)malloc(oldN * oldN * sizeof(cplx));
    cplx **regB = (cplx **)malloc(sizeof(cplx *) * oldN);

    for (int i = 0; i < N; i++) {
      A[i] = &(dataA[N * i]);
      B[i] = &(dataB[N * i]);
      if (i < oldN) {
        regA[i] = &(dataConventionalA[oldN * i]);
        regB[i] = &(dataConventionalB[oldN * i]);
      }

      for (int j = 0; j < N; j++) {
        if (i < oldN && j < oldN) {
          A[i][j] = i + j;
          B[i][j] = (i + j) * 2;

          regA[i][j] = i + j;
          regB[i][j] = (i + j) * 2;
        } else {
          A[i][j] = 0;
          B[i][j] = 0;
        }
      }
    }

    if (ProcRank == 0) {
      // Master process
      cplx *data = (cplx *)malloc(oldN * oldN * sizeof(cplx));
      cplx **strassenRes = (cplx **)malloc(sizeof(cplx *) * oldN);
      for (int i = 0; i < oldN; i++)
        strassenRes[i] = &(data[oldN * i]);

      cplx **regRes = MultiplyRegularly(regA, regB, oldN);
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Recv(&strassenRes[0][0], oldN * oldN, MPI_DOUBLE_COMPLEX, 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      bool isMatricesEqual = Compare(strassenRes, regRes, oldN);
      freeMtx(regRes);

      if (isMatricesEqual) {
        cout << "A:" << endl;
        cout << MatrixToString(regA, oldN);
        cout << endl
             << endl
             << "B:" << endl;
        cout << MatrixToString(regB, oldN);
        cout << endl
             << endl
             << "C:" << endl;
        cout << MatrixToString(strassenRes, oldN);
      } else
        cout << "Results are incorrect" << endl;
      freeMtx(strassenRes);
    } else {
      // Secondary processes

      cplx **C = Multiply(A, B, N, NewProcRank, NewProcNum, false, new_comm);
      if (NewProcRank == 0) {
        cplx **strassenRes = CropMatrixBack(C, oldN);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Send(&strassenRes[0][0], oldN * oldN, MPI_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);
        freeMtx(strassenRes);
      } else
        MPI_Barrier(MPI_COMM_WORLD);

      freeMtx(C);
    }
    freeMtx(A);
    freeMtx(B);
    freeMtx(regA);
    freeMtx(regB);

    MPI_Group_free(&new_group);
    if (MPI_COMM_NULL != new_comm)
      MPI_Comm_free(&new_comm);
  } else {
    cout << "8 processes required" << endl;
  }
  MPI_Finalize();

  return 0;
}