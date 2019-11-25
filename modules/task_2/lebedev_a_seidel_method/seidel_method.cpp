// Copyright 2019 Lebedev Alexander
#include <mpi.h>
#include <random>
#include <ctime>
#include <vector>
#include <iostream>
#include <exception>
#include "../../../modules/task_2/lebedev_a_seidel_method/seidel_method.h"

std::vector<double> getRandomVector(int n, int lowerBound, int upperBound) {
    if (n < 0) {
        throw 1;
    }

    if (upperBound < lowerBound) {
        throw 2;
    }

    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    std::vector<double> result(n);

    for (int i = 0; i < n; i++) {
        result[i] = gen() % (upperBound - lowerBound) + lowerBound;
    }

    return result;
}

std::vector<std::vector<double>> getRandomMatrix(int n, int lowerBound, int upperBound) {
    if (n < 0) {
        throw 1;
    }

    if (upperBound < lowerBound) {
        throw 2;
    }

    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    std::vector<std::vector<double>> result(n, std::vector<double>(n));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i][j] = gen() % (upperBound - lowerBound) + lowerBound;
        }
    }
    return result;
}

double vectorNorm(const std::vector<double> &v) {
    double result = 0.0;
    for (auto x : v) {
        result += x * x;
    }
    return sqrt(result);
}


// solve Ax = b, there A is a coefficientsMatrix and b is a freeMembersVector
std::vector<double> solveSequentialSeidel(std::vector<std::vector<double>> a,
                                          std::vector<double> b,
                                          double eps,
                                          int approximationsCount) {
    size_t              n = b.size();
    std::vector<double> x(n);
    std::vector<double> oldX(n, 0);

    do {
        for (size_t i = 0; i < n; i++) {
            double currentSum = b[i];

            for (size_t j = 0; j < i; j++) {
                currentSum -= a[i][j] * x[j];
            }

            for (size_t j = i + 1; j < n; j++) {
                currentSum -= a[i][j] * x[j];
            }

            oldX[i] = x[i];
            x[i] = currentSum / a[i][i];
        }
    } while (abs(vectorNorm(x) - vectorNorm(oldX)) >= eps && approximationsCount--);

    return x;
}

// solve in parallel Ax = b, there A is a coefficientsMatrix and b is a freeMembersVector
std::vector<double> solveParallelSeidel(std::vector<std::vector<double>> a,
                                          std::vector<double> b,
                                          double eps,
                                          int approximationsCount) {
    int size;
    int rank;
    int root = 0;
    int tag = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;

    int n = static_cast<int>(b.size());
    int elementsPerBlock = n / size;
    int remainElements = n % size;
    std::vector<double> x(n);
    std::vector<double> oldX(n, 0);
    std::vector<double> currentB(elementsPerBlock);
    std::vector<std::vector<double>> currentA(elementsPerBlock, std::vector<double>(n));

    MPI_Bcast(&n, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Scatter(&b[0],
                elementsPerBlock,
                MPI_DOUBLE,
                &currentB[0],
                elementsPerBlock,
                MPI_DOUBLE,
                root,
                MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            for (int j = 0; j < elementsPerBlock; j++) {
                MPI_Send(&a[i * elementsPerBlock + j],
                         n,
                         MPI_DOUBLE,
                         i,
                         tag,
                         MPI_COMM_WORLD);
            }
        }
    } else {
        for (int i = 0; i < elementsPerBlock; i++) {
            MPI_Recv(&currentA[rank * elementsPerBlock + i],
                     n,
                     MPI_DOUBLE,
                     root,
                     tag,
                     MPI_COMM_WORLD,
                     &status);
        }
    }

    do {
        for (int i = 0; i < n; i++) {
            oldX[i] = x[i];
        }

        // Calculation process for main blocks

        for (int i = 0; i < elementsPerBlock; i++) {
            int currInd = i + elementsPerBlock * rank;

            x[currInd] = b[currInd];

            for (int j = currInd + 1; j < n; j++) {
                x[currInd] -= x[j] * a[currInd][j];
            }
        }

        for (int j = 0; j < rank * elementsPerBlock; j++) {
            int indexOfBlock = j / elementsPerBlock % size;

            MPI_Recv(&x[j], 1, MPI_DOUBLE, indexOfBlock, tag, MPI_COMM_WORLD, &status);
            for (int i = 0; i < elementsPerBlock; i++) {
                int currInd = i + elementsPerBlock * rank;
                x[currInd] -= x[j] * (a[currInd][j]);
            }
        }

        for (int i = 0; i < elementsPerBlock; i++) {
            int currInd = i + elementsPerBlock * rank;
            for (int j = elementsPerBlock * rank; j < currInd; j++) {
                x[currInd] -= x[j] * (a[currInd][j]);
            }

            x[currInd] /= a[currInd][currInd];

            for (int j = currInd + 1; j < size; j++) {
                MPI_Send(&x[currInd], 1, MPI_DOUBLE, j, tag, MPI_COMM_WORLD);
            }
            if (rank != root) MPI_Send(&x[currInd], 1, MPI_DOUBLE, root, tag, MPI_COMM_WORLD);
        }

        // Calculation process for remain Elements (run under root because it will be first free proc )

        if (rank == root) {
            for (int i = n - remainElements; i < n; i++) {
                x[i] = b[i];

                for (int j = i + 1; j < n; j++) {
                    x[i] -= x[j] * a[i][j];
                }
            }
            for (int j = 0; j < n - remainElements; j++) {
                int indexOfBlock = j / elementsPerBlock % size;

                if (indexOfBlock != root) {
                    MPI_Recv(&x[j], 1, MPI_DOUBLE, indexOfBlock, tag, MPI_COMM_WORLD, &status);
                }

                for (int i = n - remainElements; i < n; i++) {
                    x[i] -= x[j] * (a[i][j]);
                }
            }

            for (int i = n - remainElements; i < n; i++) {
                for (int j = n - remainElements; j < i; j++) {
                    x[i] -= x[j] * (a[i][j]);
                }
                x[i] /= a[i][i];
            }
        }
    } while (abs(vectorNorm(x) - vectorNorm(oldX)) > eps);
    return x;
}
