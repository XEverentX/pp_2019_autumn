// Copyright 2019 Lebedev Alexander
#include <mpi.h>
#include <random>
#include <ctime>
#include <vector>
#include <iostream>
#include <algorithm>
#include "../../../modules/task_2/lebedev_a_seidel_method/seidel_method.h"

std::vector<double> getRandomVector(int n, int lowerBound, int upperBound) {
    std::random_device rd;
    std::mt19937 rnd(rd());
    std::uniform_int_distribution<int> gen(lowerBound, upperBound);

    std::vector<double> result(n);

    for (int i = 0; i < n; i++) {
        result[i] = gen(rnd);
    }

    return result;
}

std::vector<double> getRandomMatrix(int n, int lowerBound, int upperBound) {
    std::random_device rd;
    std::mt19937 rnd(rd());
    std::uniform_int_distribution<int> gen(lowerBound, upperBound);

    std::vector<double> result(n * n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i * n + j] = gen(rnd);
        }
    }
    return result;
}

void transpanent(std::vector<double> &v, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            std::swap(v[i * n + j], v[j * n + i]);
        }
    }
}

// solve Ax = b, there A is a coefficientsstd::vector<double> and b is a freeMembersVector
std::vector<double> solveSequentialSeidel(std::vector<double> a,
                                          std::vector<double> b,
                                          double eps,
                                          int approximationsCount) {
    int                 n = static_cast<int>(b.size());
    std::vector<double> x(n);
    double              currEps = 0.;
    do {
        currEps = 0.;
        for (int i = 0; i < n; i++) {
            double currentSum = b[i];

            for (int j = 0; j < i; j++) {
                currentSum -= a[i * n + j] * x[j];
            }

            for (int j = i + 1; j < n; j++) {
                currentSum -= a[i * n + j] * x[j];
            }

            double newX = currentSum / a[i* n + i];
            currEps = std::max(std::fabs(newX - x[i]), currEps);
            x[i] = newX;
        }
    } while (currEps >= eps && approximationsCount--);

    return x;
}

// solve in parallel Ax = b, there A is a coefficientsstd::vector<double> and b is a freeMembersVector
std::vector<double> solveParallelSeidel(std::vector<double> a,
                                        std::vector<double> b,
                                        double eps,
                                        int approximationsCount,
                                        MPI_Comm comm) {
    int size;
    int rank;
    int root = 0;

    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    int n = 0;

    if (rank == root) {
        n = static_cast<int>(b.size());
    }

    MPI_Bcast(&n, 1, MPI_INT, root, comm);

    int    tag              = 0;
    int    elementsPerBlock = n / size;
    int    remainElements   = n % size + elementsPerBlock;
    int    elementsCount    = (rank ==  size - 1 ? remainElements : elementsPerBlock);
    int    indexOfProces    = (size <= n ? 0 : size - 1);
    double currEps          = 0.;
    double currSum          = 0.;
    double globSum          = 0.;

    std::vector<double> x;
    std::vector<double> currX(elementsCount);
    std::vector<double> currA(elementsCount * n);

    // Init data on all Proc
    if (rank == root) {
        x.resize(n);
        transpanent(a, n);

        for (int pr = 1; pr < size; pr++) {
            int startIndex = elementsPerBlock * pr;
            int destCount  = (pr == size - 1 ? remainElements : elementsPerBlock);
            if (destCount) {
                MPI_Send(&a[startIndex * n], destCount * n, MPI_DOUBLE, pr, tag, comm);
            }
        }
        if (elementsCount) {
            currA.assign(a.begin(), (a.begin() + n * elementsCount));
        }
    } else {
        if (elementsCount) {
            MPI_Recv(&currA[0], elementsCount * n, MPI_DOUBLE, root, tag, comm, MPI_STATUS_IGNORE);
        }
    }
    
    // Execution parth
    do {
        indexOfProces = (size <= n ? 0 : size - 1);
        for (int i = 0; i < n; i++) {
            currSum = 0.;
            globSum = 0.;
            for (int j = 0; j < elementsCount; j++) {
                currSum -= currA[j * n + i] * currX[j];
            }

            MPI_Reduce(&currSum, &globSum, 1, MPI_DOUBLE, MPI_SUM, root, comm);

            int indexOfElement = i - indexOfProces * elementsPerBlock;

            if (rank == root) {
                auto newX = (b[i] + globSum + x[i] * a[i * n + i]) / a[i * n + i];
                currEps = std::max(std::fabs(newX - x[i]), currEps);
                x[i] = newX;
            }

            if (indexOfProces != root) {
                if (rank == root) {
                    MPI_Send(&x[i], 1, MPI_DOUBLE, indexOfProces, tag, comm);
                }
                if (indexOfProces == rank) {
                    MPI_Recv(&currX[indexOfElement], 1, MPI_DOUBLE, root, tag, comm, MPI_STATUS_IGNORE);
                }
            } else {
                if (rank == root && indexOfProces == root) {
                    currX[indexOfElement] = x[i];
                }
            }
            if (elementsPerBlock && (i + 1) % elementsPerBlock == 0 && indexOfProces != size - 1) {
                indexOfProces++;
            }
        }

        MPI_Bcast(&currEps, 1, MPI_DOUBLE, root, comm);
    } while (currEps >= eps && approximationsCount--);
    return x;
}
