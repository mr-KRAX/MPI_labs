#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <sstream>
#include "mpi.h"
#include <time.h>
#include <string>

using namespace std;

const int MAX_LINT_LEN = 26;


/**
 * @brief Logging
 */
bool INFO_logOn = true;
bool DEBUG_logOn = false;
#define INFO 0
#define DEBUG 1
#define LOG(group, msg, ...) do{\
  if (INFO_logOn && group == INFO  ||\
			DEBUG_logOn && group == DEBUG) {\
    printf(msg"\n", __VA_ARGS__);\
  }\
} while (0)

/**
 * @brief convert lInt number to string
 * 
 * @param num 
 * @param size 
 * @return string 
 */
string lInt2str(int* num, int size = MAX_LINT_LEN) {
  stringstream ss;
  bool start_flag = false;
  for (int i = size - 1; i >= 0; i--) {
    if (!start_flag && num[i] == 0 && i != 0)
      continue;
    start_flag = true;
    ss << (int)num[i] << (i % 3 == 0 && i != 0 ? "." : "");
  }
  return ss.str();
}

/**
 * @brief return new lInt number initialized with 0 
 * 
 * @param size 
 * @return int* zero number
 */
int* zeroLint(int size = MAX_LINT_LEN) {
  int* res = new int[size];
  for (int i = 0; i < size; i++)
    res[i] = 0;
  return res;
}

int* int2lInt(int n, int size = MAX_LINT_LEN) {
  int* res = zeroLint();
  for (int i = 0; n != 0 && i < size; i++){
    res[i] = n % 10;
    n /= 10;
  }
  return res;
}

/**
 * @brief Generate random lInt number with a specified length 
 * 
 * @param size 
 * @return int* generated number
 */
int* randLInt(int size = MAX_LINT_LEN) {
  int* res = zeroLint();
  for (int i = 0; i < size; i++)
    res[i] = rand() % 10;
  return res;
}

/**
 * @brief fix lInt number after multiplication
 * 
 * @param a 
 * @param size 
 */
void fixLInt(int* a, int size = MAX_LINT_LEN) {
  for (int i = 0; i < size; i++)
    if (a[i] > 9) {
      a[i + 1] += a[i] / 10;
      a[i] %= 10;
    }
}

/**
 * @brief multiply two lInt numbers
 * 
 * @param a 
 * @param b 
 * @param aSize 
 * @param bSize 
 * @return int* result lInt number
 */
int* multiply(int* a, int* b, int aSize = MAX_LINT_LEN, int bSize = MAX_LINT_LEN) {
  int* res = zeroLint();
  for (int bi = 0; bi < bSize; bi++) {
    int* tmp = zeroLint(aSize + 1);
    for (int ai = 0; ai < aSize; ai++)
      tmp[ai] = a[ai] * b[bi];
    fixLInt(tmp, aSize + 1);
    for (int i = 0; i < aSize + 1; i++)
      res[bi + i] = res[bi + i] + tmp[i];
    delete[] tmp;
  }
  fixLInt(res);
  return res;
}


int numbers[] = {34, 92, 73, 45, 43, 47, 82, 72, 94, 95, 39, 57, 70, 50};
/**
 * @brief Master process task
 * @param proc_num the number of slaves initialized
*/
void masterTask(int proc_num) {
  LOG(INFO, "-Master: started");
  MPI_Datatype lInt;
  MPI_Type_contiguous(MAX_LINT_LEN, MPI_INT, &lInt);
  MPI_Type_commit(&lInt);

  for (int i = 0; i < 1; i++) {
    for (int rank_i = 1; rank_i < proc_num; rank_i++) {
      int* num1 = int2lInt(numbers[2 * (rank_i - 1)]);
      int* num2 = int2lInt(numbers[2*(rank_i - 1) + 1]);
      MPI_Send(num1, 1, lInt, rank_i, 0, MPI_COMM_WORLD);
      MPI_Send(num2, 1, lInt, rank_i, 0, MPI_COMM_WORLD);

      LOG(DEBUG, "-Master: numbers %s and %s sent to proc %d", lInt2str(num1).c_str(), lInt2str(num2).c_str(), rank_i);

      delete[] num1;
      delete[] num2;
    }
    LOG(INFO, "-Master: Numbers for all slaves sent (iteration %d)", i);
    int* ans = int2lInt(1);
    for (int rank_i = 1; rank_i < proc_num; rank_i++) {
      int* res = new int[MAX_LINT_LEN];
      MPI_Status status;
      MPI_Recv(res, 1, lInt, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      int* tmp = ans;
      ans = multiply(res, ans);
      LOG(DEBUG, "-Master: from proc %d: res: %s, curr total: %s", status.MPI_SOURCE, lInt2str(res).c_str(), lInt2str(ans).c_str());
      delete[] res;
      delete[] tmp;
    }
    LOG(INFO, "-Master: Answers from all slaves received (iteration %d), total: %s", i, lInt2str(ans).c_str());
    delete[] ans;
  }
  MPI_Type_free(&lInt);
  return;
}

/**
 * @brief Slave process task
 * @param rank The rank of the current slave process
*/
void slaveTask(int rank) {
  LOG(DEBUG, "Slave %d: stared", rank);
  MPI_Datatype lInt;
  MPI_Type_contiguous(MAX_LINT_LEN, MPI_INT, &lInt);
  MPI_Type_commit(&lInt);
  MPI_Status status;

  int* num1 = new int[MAX_LINT_LEN];
  int* num2 = new int[MAX_LINT_LEN];
  for (int i = 0; i < 1; i++) {
  //while (true) {
    MPI_Recv(num1, 1, lInt, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(num2, 1, lInt, 0, 0, MPI_COMM_WORLD, &status);
    LOG(DEBUG, "Slave %d: numbers %s and %s received", rank, 
      lInt2str(num1).c_str(),
      lInt2str(num2).c_str());

    int* res = multiply(num1, num2);
    MPI_Send(res, 1, lInt, 0, 0, MPI_COMM_WORLD);
    LOG(INFO, "Slave %d: math done: %s * %s = %s", rank, 
      lInt2str(num1).c_str(), 
      lInt2str(num2).c_str(), 
      lInt2str(res).c_str());
    delete[] res;
  }
  LOG(DEBUG, "Slave %d: finished", rank);
  MPI_Type_free(&lInt);
  delete[] num1;
  delete[] num2;
  return;
}

int main(int* argc, char** argv) {
  srand(time(0));
  MPI_Init(argc, &argv);

  int proc_num, rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

  LOG(DEBUG, "Process %d initialized (total %d)", rank, proc_num);

  if (rank == 0)
    masterTask(proc_num);
  else
    slaveTask(rank);

  MPI_Finalize();
  return 0;
}