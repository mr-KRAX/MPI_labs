#include <cstdio>
#include "mpi.h"

using namespace std;

const int M = 5; // The number of iterations

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
 * @brief Master process task
 * @param proc_num the number of slaves initialized
*/
void masterTask(int proc_num) {
  printf("Master: started\n");

  for (int i = 0; i < M; i++) {
    int msg = i;
    MPI_Bcast(&msg, 1, MPI_INT, 0, MPI_COMM_WORLD);
    LOG(INFO, "Master: message %d sent by bcast", msg);
    int msg_sum;
    int tmp_buf = 0;
    MPI_Reduce(&tmp_buf, &msg_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    LOG(INFO, "Master: Replies from all slaves received (iteration %d, sum %d)", i, msg_sum);
  }
  int	finish_msg = -1;
  MPI_Bcast(&finish_msg, 1, MPI_INT, 0, MPI_COMM_WORLD);
  LOG(INFO, "Master: finish signal sent");
  return;
}

/**
 * @brief Slave process taks
 * @param rank The rank of the current slave process
*/
void slaveTask(int rank) {
  LOG(INFO, "Slave %d: stared", rank);
  //for (int i = 0; i < M+1; i++){
  while (true) {
    int income_msg;
    MPI_Status status;
    MPI_Bcast(&income_msg, 1, MPI_INT, 0, MPI_COMM_WORLD);
    LOG(DEBUG, "Slave %d: Message %d received", rank, income_msg);
    if (income_msg == -1) {
      LOG(DEBUG, "Slave %d: finish signal received", rank);
      break;
    }
    MPI_Reduce(&income_msg, nullptr, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    LOG(DEBUG, "Slave %d: Reply sent", rank);
  }
  LOG(INFO, "Slave %d: finished", rank);
  return;
}


int main(int* argc, char** argv) {
  MPI_Init(argc, &argv);
  int proc_num, rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

  LOG(INFO, "Process %d initialized (total %d)", rank, proc_num);


  if (rank == 0) {
    double t1, t2, dt;
    t1 = MPI_Wtime();
    masterTask(proc_num);
    t2 = MPI_Wtime();
    dt = t2 - t1;
    LOG(INFO, "Time: %lf", dt);
  }
  else {
    slaveTask(rank);
    MPI_Finalize();
    return 0;
  }

  MPI_Finalize();
  return 0;
}