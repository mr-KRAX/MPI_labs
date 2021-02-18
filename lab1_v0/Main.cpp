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
		for (int rank_i = 1; rank_i < proc_num; rank_i++) {
			MPI_Send(&msg, 1, MPI_INT, rank_i, 0, MPI_COMM_WORLD);
			LOG(DEBUG, "Master: message %d sent to proc %d", msg, rank_i);
		}
		LOG(INFO, "Master: Messages for all slaves sent (iteration %d)", i);
		for (int rank_i = 1; rank_i < proc_num; rank_i++) {
			int income_msg;
			MPI_Status status;
			MPI_Recv(&income_msg, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			LOG(DEBUG, "Master: reply %d received from proc %d", income_msg, status.MPI_SOURCE);
		}
		LOG(INFO, "Master: Relies from all slaves received (iteration %d)", i);
	}
	int	finish_msg = -1;
	for (int rank_i = 1; rank_i < proc_num; rank_i++) {
		MPI_Send(&finish_msg, 1, MPI_INT, rank_i, 0, MPI_COMM_WORLD);
	}
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
	while (true){
		int income_msg;
		MPI_Status status;
		MPI_Recv(&income_msg, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		LOG(DEBUG, "Slave %d: Message %d received", rank, income_msg);
		if (income_msg == -1) {
			LOG(DEBUG, "Slave %d: finish signal received", rank);
			break;
		}
		MPI_Send(&income_msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		LOG(DEBUG, "Slave %d: Reply sent", rank);
	}
	LOG(INFO, "Slave %d: finished", rank);
	return;
}


int main(int *argc, char **argv) {
	MPI_Init(argc, &argv);
	int proc_num, rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

	LOG(INFO, "Process %d initialized (total %d)", rank, proc_num);

	if (rank == 0)
		masterTask(proc_num);
	else
		slaveTask(rank);

	MPI_Finalize();
	return 0;
} 