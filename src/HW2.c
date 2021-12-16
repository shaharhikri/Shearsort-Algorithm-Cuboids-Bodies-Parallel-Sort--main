#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "mpi.h"

#define CART_DIM  2
int MATRIX_DIM = 7;
int MATRIX_CELLS_COUNT = 49;

typedef enum {
	COLLS = 0, ROWS = 1,
} MatrixPassDirection;

typedef enum {
	RECEIVING = 0, SENDING = 1, NO_COMM = -1
} CommDirection;

typedef enum {
	ASCENDING = 0, DESCENDING = 1
} SortDirection;

typedef enum {
	false = 0, true = 1
} bool;

void ShearSort(float *myCuboid, MPI_Comm comm);
void OddEvenSort(int *coord, float *myCuboid, MatrixPassDirection passDirection,
		MPI_Comm comm);
void ExchangeBetweenNeighbors(float *myCuboid, CommDirection commDirection,
		SortDirection sortDirection, int neighborRank, MPI_Comm comm);
float getTotSurface(float *cuboid);
bool isGreater(float *cuboid1, float *cuboid2, SortDirection sortDirection);
CommDirection GetCommDirection(int *coord, int iteration,
		MatrixPassDirection direction);
SortDirection GetSortDirection(int *coord, MatrixPassDirection direction);
void printArray(float **matrix);
void printMatrix(float **matrix);
void printCuboid(float *cuboid);
void matrixToArray(float **matrix);
void cuboidCpy(float *to_cuboid, float *from_cuboid);
void swapCuboid(float *from_cuboid, float *to_cuboid);
void matrixToFlatted(float **matrix, float *flattedMatrix);
void flattedToMatrix(float *flattedMatrix, float **matrix);
void readDataFromFIle(char *filename, float ***matrix, float **flatted_matrix,
		int myRank, int numberOfWorkers, int *Ascending_p);
void writeToResultFIle(char *filename, float **matrix, float *flatted_matrix,
		int myRank, int Ascending);

/* 'cuboids.dat' and 'result.dat' is in the subdirectory of src dir */
int main(int argc, char *argv[]) {
	int numberOfWorkers, myRank;
	float **matrix;
	float *flatted_matrix;
	float myCuboid[4]; //int receivedNum;
	int periods[] = { 0, 0 }; // we don't want ciclic topolgy
	int i;
	int Ascending = 0;

	MPI_Comm newComm;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &numberOfWorkers);

	readDataFromFIle("../cuboids.dat", &matrix, &flatted_matrix, myRank,
			numberOfWorkers, &Ascending);

	int dims[] = { MATRIX_DIM, MATRIX_DIM };

	// Distribute CELLS_PER_PROC==1 elements to each node
	MPI_Scatter(flatted_matrix, 4, MPI_FLOAT, myCuboid, 4, MPI_FLOAT, 0,
			MPI_COMM_WORLD);
	// Create MPI Cartesian topology with 2 dimmentions
	MPI_Cart_create(MPI_COMM_WORLD, CART_DIM, dims, periods, 0, &newComm);
	// Sort matrix. Complexity O(2*log2(n)+1)
	ShearSort(myCuboid, newComm);
	// Receive sorted data
	MPI_Gather(myCuboid, 4, MPI_FLOAT, flatted_matrix, 4, MPI_FLOAT, 0,
			MPI_COMM_WORLD);

	writeToResultFIle("..//result.dat ", matrix, flatted_matrix, myRank,
			Ascending);

	if (myRank == 0) {
		//printMatrix(matrix); //printing the fixed matrix - for check only
		/* free matrix&flatted_matrix mem*/
		free(flatted_matrix);
		for (i = 0; i < MATRIX_CELLS_COUNT; i++)
			free(matrix[i]);
		free(matrix);
	}
	MPI_Finalize();

	return 0;
}

void ShearSort(float *myCuboid, MPI_Comm comm) {
	int rank;
	int coord[2];
	MPI_Comm_rank(comm, &rank);
	MPI_Cart_coords(comm, rank, CART_DIM, coord);

	int totalIterations = (int) ceil(log2((double) MATRIX_DIM)) + 1;

	for (int i = 0; i <= totalIterations; i++) {
		// Rows pass
		OddEvenSort(coord, myCuboid, ROWS, comm); //
		// Columns pass
		OddEvenSort(coord, myCuboid, COLLS, comm); //
	}
}

void OddEvenSort(int *coord, float *myCuboid, MatrixPassDirection passDirection,
		MPI_Comm comm) {
	int neighbor1, neighbor2, neighborRankForExchange;
	CommDirection commDirection;
	SortDirection sortDirection = GetSortDirection(coord, passDirection);
	MPI_Cart_shift(comm, passDirection, 1, &neighbor1, &neighbor2);

	for (int i = 0; i < MATRIX_DIM; i++) {
		commDirection = GetCommDirection(coord, i, passDirection);
		neighborRankForExchange =
				commDirection == SENDING ? neighbor2 : neighbor1;

		if (neighborRankForExchange != MPI_PROC_NULL) // Exchange only if we in bounds
			ExchangeBetweenNeighbors(myCuboid, commDirection, sortDirection,
					neighborRankForExchange, comm); //
	}
}

void ExchangeBetweenNeighbors(float *myCuboid, CommDirection commDirection,
		SortDirection sortDirection, int neighborRank, MPI_Comm comm) {
	MPI_Status status;
	float result[4];
	float received[4];
	if (commDirection == SENDING) // Sending side
			{
		MPI_Send(myCuboid, 4, MPI_FLOAT, neighborRank, 0, comm);
		MPI_Recv(result, 4, MPI_FLOAT, neighborRank, 0, comm, &status);

		cuboidCpy(myCuboid, result);
	} else // Receiving side. Make the check/sort
	{
		MPI_Recv(received, 4, MPI_FLOAT, neighborRank, 0, comm, &status);

		if (isGreater(myCuboid, received, sortDirection))
			swapCuboid(received, myCuboid);

		MPI_Send(received, 4, MPI_FLOAT, neighborRank, 0, comm);
	}
}

float getTotSurface(float *cuboid) {
	return 2
			* (cuboid[1] * cuboid[2] + cuboid[1] * cuboid[3]
					+ cuboid[2] * cuboid[3]);
}

bool isGreater(float *cuboid1, float *cuboid2, SortDirection sortDirection) {
	float surfaceArea1 = getTotSurface(cuboid1);
	float surfaceArea2 = getTotSurface(cuboid2);

	return sortDirection == ASCENDING ?
			surfaceArea1 < surfaceArea2 : surfaceArea1 > surfaceArea2;
}

/* if position even and iteration even we are sending, otherwise receiving
 if position odd and iteration odd we are sending, otherwise receiving*/
CommDirection GetCommDirection(int *coord, int iteration,
		MatrixPassDirection direction) {
	return (iteration % 2 == coord[direction] % 2) ? SENDING : RECEIVING;
}

/* Even Row  ASCENGING, Odd Row   DESCENDING, * Coll always ASCENGING */
SortDirection GetSortDirection(int *coord, MatrixPassDirection direction) {
	if (direction == COLLS)
		return ASCENDING;
	return (SortDirection) (coord[0] % 2);
}

/* print cuboid matrix as array (1 line)*/
void printArray(float **matrix) {
	int i;
	for (i = 0; i < MATRIX_CELLS_COUNT; i++)
		printCuboid(matrix[i]); //printf("%d ",matrix[i]);
	printf("\n");
}

/* print cuboid matrix*/
void printMatrix(float **matrix) {
	int i;
	for (i = 0; i < MATRIX_CELLS_COUNT; i++) {
		printCuboid(matrix[i]); //printf("%d ",matrix[i]);
		if ((i + 1) % MATRIX_DIM == 0)
			printf("\n");
	}
}

/* print cuboid*/
void printCuboid(float *cuboid) {
	int i;
	printf("[%.0f|", cuboid[0]);
	for (i = 1; i < 4; i++)
		printf("%.1f ", cuboid[i]);
	printf("- %.2f] ", getTotSurface(cuboid));
}

/* fix the matrix by reversing every odd row - turning 'matrix' into a 1 dim array(after sorting) */
void matrixToArray(float **matrix) {
	float *row[4];

	int i, j;

	for (i = 1; i < MATRIX_DIM; i += 2) {
		for (j = 0; j < MATRIX_DIM; j++)
			row[MATRIX_DIM - j - 1] = matrix[i * MATRIX_DIM + j];
		for (j = 0; j < MATRIX_DIM; j++)
			matrix[i * MATRIX_DIM + j] = row[j];
	}
}

/* copy 'from_cuboid' to 'to_cuboid'*/
void cuboidCpy(float *to_cuboid, float *from_cuboid) {
	int i;
	for (i = 0; i < 4; i++)
		to_cuboid[i] = from_cuboid[i];
}

/*swap cubids*/
void swapCuboid(float *from_cuboid, float *to_cuboid) {
	float temp[4];
	cuboidCpy(temp, from_cuboid);
	cuboidCpy(from_cuboid, to_cuboid);
	cuboidCpy(to_cuboid, temp);
}

/* convert cuboids array to int array*/
void matrixToFlatted(float **matrix, float *flattedMatrix) {
	int i, j;
	for (i = 0; i < MATRIX_CELLS_COUNT; i++) {
		for (j = 0; j < 4; j++)
			flattedMatrix[4 * i + j] = matrix[i][j];
	}
}

/* convert int array to cuboids array*/
void flattedToMatrix(float *flattedMatrix, float **matrix) {
	int i, j;
	for (i = 0; i < MATRIX_CELLS_COUNT; i++) {
		for (j = 0; j < 4; j++)
			matrix[i][j] = flattedMatrix[4 * i + j];
	}
}

/* read data and cuboids from file */
void readDataFromFIle(char *filename, float ***matrix, float **flatted_matrix,
		int myRank, int numberOfWorkers, int *Ascending_p) {
	if (myRank == 0) {
		int i, j;
		FILE *f = fopen(filename, "r");
		if (!f) {
			fprintf(stderr, "File '%s' open Error\n", filename);
			perror("Error printed by perror\n");
		}
		fscanf(f, "%d%d", &MATRIX_CELLS_COUNT, Ascending_p);

		if (numberOfWorkers != MATRIX_CELLS_COUNT) {
			fprintf(stderr, "Program requires %d nodes\n", MATRIX_CELLS_COUNT);
			MPI_Abort(MPI_COMM_WORLD, 0);
		}

		(*matrix) = (float**) malloc(MATRIX_CELLS_COUNT * sizeof(float*));
		(*flatted_matrix) = (float*) malloc(
				4 * MATRIX_CELLS_COUNT * sizeof(float));

		for (i = 0; i < MATRIX_CELLS_COUNT; i++) {
			(*matrix)[i] = (float*) malloc(4 * sizeof(float)); //new Cubid
			for (j = 0; j < 4; j++)
				fscanf(f, "%f", &((*matrix)[i][j]));
		}

		fclose(f);
		matrixToFlatted(*matrix, *flatted_matrix);
	}

	MPI_Bcast(&MATRIX_CELLS_COUNT, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(Ascending_p, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MATRIX_DIM = (int) sqrt(MATRIX_CELLS_COUNT);
}

/* fix matrix to be sorted as it should be(after shearsort) and write it's cuboids id's to file in Ascending/Descending order*/
void writeToResultFIle(char *filename, float **matrix, float *flatted_matrix,
		int myRank, int Ascending) {
	if (myRank == 0) {
		flattedToMatrix(flatted_matrix, matrix);
		matrixToArray(matrix);
		FILE *f = fopen(filename, "w");
		if (!f) {
			fprintf(stderr, "File '%s' open Error\n", filename);
			perror("Error printed by perror\n");
		}

		int i;

		if (Ascending) {
			for (i = 0; i < MATRIX_CELLS_COUNT; i++)
				fprintf(f, "%.0f ", matrix[i][0]);
		} else {
			for (i = MATRIX_CELLS_COUNT - 1; i >= 0; i--)
				fprintf(f, "%.0f ", matrix[i][0]);
		}

		fclose(f);
	}
}

