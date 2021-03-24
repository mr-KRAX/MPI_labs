#include <cstdio>
#include <cstring>>
#include "mpi.h"

using namespace std;

const int M = 5; // The number of iterations
int proc_num;

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
  int* msg_out = new int[proc_num - 1];
  int* msg_in = new int[proc_num-1];
  int* tmp_buf = new int[proc_num-1];

  for (int i = 0; i < M; i++) {
    memset(msg_out, i, proc_num * sizeof(int)); // Так и задумано, чтобы циклы не делать
    MPI_Scatter(msg_out, 1, MPI_INT, tmp_buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    LOG(INFO, "Master: message %d sent by bcast", msg_out[0]);
    MPI_Gather(tmp_buf, 1, MPI_INT, msg_in, 1, MPI_INT, 0, MPI_COMM_WORLD);
    LOG(INFO, "Master: Replies from all slaves received (iteration %d, sum %d)", i, msg_in[0]);
  }
  memset(msg_out, -1, proc_num * sizeof(int)); // Так и задумано, все работает, так как -1 = fff...
  MPI_Scatter(msg_out, 1, MPI_INT, tmp_buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
  LOG(INFO, "Master: finish signal sent");
  return;
}

/**
 * @brief Slave process taks
 * @param rank The rank of the current slave process
*/
void slaveTask(int rank) {
  LOG(INFO, "Slave %d: stared", rank);
  int msg_in;
  int msg_out;
  int* tmp_buf = new int[proc_num - 1];



  //for (int i = 0; i < M+1; i++){
  while (true) {
    MPI_Status status;
    MPI_Scatter(tmp_buf, 1, MPI_INT, &msg_in, 1, MPI_INT, 0, MPI_COMM_WORLD);
    LOG(DEBUG, "Slave %d: Message %d received", rank, msg_in);
    if (msg_in == -1) {
      LOG(DEBUG, "Slave %d: finish signal received", rank);
      break;
    }
    msg_out = msg_in;
    MPI_Gather(&msg_out, 1, MPI_INT, tmp_buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    LOG(DEBUG, "Slave %d: Reply sent", rank);
  }
  LOG(INFO, "Slave %d: finished", rank);
  return;
}


int main(int* argc, char** argv) {
  MPI_Init(argc, &argv);
  int rank;
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