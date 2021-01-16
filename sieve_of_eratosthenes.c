#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <sys/sysinfo.h>
#include <mpi.h>

#define MASTER 0

typedef struct {
	unsigned start, end, max_root;
}compute_t;

void compute_primes_sequentially(unsigned k, const unsigned max, bool *list)
{
	unsigned i, multiplier = 2;

	while (k * k <= max) {
		/* strike-out the multiples of k */
		while ((k * multiplier) <= max) {
			list[k * multiplier] = false;
			++multiplier;
		}

		/* reset multiplier */
		multiplier = 2;

		/* determine the lowest un-marked k-value */
		for (i = k + 1; i <= max; i++) {
			if (list[i]) {
				k = i;
				break;
			}
		}
	}
}

void bounded_strike(const unsigned lowest_prime, compute_t *job, bool *const list)
{
	unsigned lowest_multiplier, val_to_strike;

	if (job->start % lowest_prime)
		lowest_multiplier = (job->start / lowest_prime) + 1;
	else
		lowest_multiplier = job->start / lowest_prime;

	while ((val_to_strike = lowest_prime * lowest_multiplier) <= job->end) {
		list[val_to_strike] = false;
		lowest_multiplier++;
	}
}

void compute_on_chunk(compute_t *job, bool *const list)
{
	unsigned lowest_prime;

	/* find the lowest prime that was computed sequentially */
	for (lowest_prime = 2; lowest_prime <= job->max_root; lowest_prime++)
		if (list[lowest_prime])
			bounded_strike(lowest_prime, job, list);
}

int main(int argc, char **argv)
{
	unsigned i = 0, buf_size, max, max_root, k = 2, cores, remaining, chunk_sz, start;
	compute_t *chunk, *param;
	double starttime, endtime;
	bool *list, *final_list, *seeds;

	MPI_Init(NULL, NULL);

	/* determine world size */
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	/* determine my rank */
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	if (argc != 2) {
		printf("Usage: %s <Max Value>\n", argv[0]);
		MPI_Abort(MPI_COMM_WORLD, -1);
		exit(EXIT_FAILURE);
	}

	if ((max = atoi(argv[1])) <= 2) {
		printf("Max should be an integer value > 2\n");
		MPI_Abort(MPI_COMM_WORLD, -1);
		exit(EXIT_FAILURE);
	}

	cores = world_size;
	buf_size = max + 1;

	/* create a list of integers upto max */
	list = (bool *)malloc(buf_size * sizeof(bool));
	if (list == NULL) {
		perror("malloc");
		MPI_Abort(MPI_COMM_WORLD, -1);
		exit(EXIT_FAILURE);
	}
	
	final_list = malloc(buf_size * sizeof(bool));
	if (final_list == NULL) {
		perror("malloc");
		MPI_Abort(MPI_COMM_WORLD, -1);
		exit(EXIT_FAILURE);
	}
	
	memset(list, true, (max + 1) * sizeof(bool));

	/* compute the root of max(rounded to the nearest integer) */
	max_root = nearbyint(sqrt(max));

	if (world_rank == MASTER) {
		seeds = (bool *)malloc(buf_size * sizeof(bool));
		if (seeds == NULL) {
			perror("malloc");
			MPI_Abort(MPI_COMM_WORLD, -1);
			exit(EXIT_FAILURE);
		}
		/* initialize the array to true */
		memset(seeds, true, (max + 1) * sizeof(bool));
		
		/* sequentially compute primes upto the square root of max */
		compute_primes_sequentially(k, max_root, seeds);
		
		/* broadcast the seeds to all other processes */
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(seeds, buf_size, MPI_C_BOOL, MASTER, MPI_COMM_WORLD);
	}

	/* receive the seeds from the master */
	if (world_rank != MASTER) {
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(list, buf_size, MPI_C_BOOL, MASTER, MPI_COMM_WORLD);
	}

	/* dynamically allocate memory for the structure */
	chunk_sz = (max - max_root) % cores ? ((max - max_root) / cores) + (unsigned)1 : (max - max_root) / cores;
	chunk = malloc(cores * sizeof(compute_t));
	if (chunk == NULL) {
		perror("malloc");
		MPI_Abort(MPI_COMM_WORLD, -1);
		exit(EXIT_FAILURE);
	}

	param = malloc(sizeof(compute_t));
	if (param == NULL) {
		perror("malloc");
		MPI_Abort(MPI_COMM_WORLD, -1);
		exit(EXIT_FAILURE);
	}

	start = max_root;
	remaining = max - max_root;

	/* populate the structures */
	for (i = 0; i < cores && remaining; i++) {
		if (remaining < chunk_sz)
			chunk_sz = remaining;

		chunk[i].start = start + 1;
		chunk[i].end = start + chunk_sz;
		chunk[i].max_root = max_root;

		start += chunk_sz;
		remaining -= chunk_sz;
	}

	/* scatter the chunk parameters across all the processes */
	MPI_Scatter(chunk, 3, MPI_UNSIGNED, param, 3, MPI_UNSIGNED, MASTER, MPI_COMM_WORLD);

	/* start measuring the time */
	starttime = MPI_Wtime();

	/* work on your chunk */
	compute_on_chunk(param, list);

	/* compute the time taken to finish execution */
	MPI_Barrier(MPI_COMM_WORLD);
	endtime = MPI_Wtime();
	if (world_rank == MASTER)
		printf("Time taken to compute: %f second(s)\n", endtime - starttime);

	/* reduce the "list" to "final list" */
	MPI_Reduce(list, final_list, buf_size, MPI_C_BOOL, MPI_LAND, MASTER, MPI_COMM_WORLD);

	//Uncomment the lines below, if you wish to see the output of the program
	/*
	if (world_rank == MASTER) {
		for (i = k; i <= max; i++)
			if (final_list[i])
				printf("%d\n", i);
	}
	*/
	
	/* release the dynamically allocated memory */
	free(chunk);
	free(list);
	free(final_list);
	if (world_rank == MASTER)
		free(seeds);

	MPI_Finalize();

	return 0;
}
