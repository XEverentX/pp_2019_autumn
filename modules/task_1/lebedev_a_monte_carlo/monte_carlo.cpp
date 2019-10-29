// Copyright 2019 Lebedev Alexander

#include <mpi.h>
#include <iostream>
#include <random>
#include <ctime>
#include <numeric>
#include <vector>
#include <algorithm>
#include <functional>
#include "../../../modules/task_1/lebedev_a_monte_carlo/monte_carlo.h"

lambda getUniformDistributionDensity(double lowBoundary,
                                     double highBoundary)
{
    return [=] (double x) -> double
    {
        return 1. / (highBoundary - lowBoundary);
    };
}

generator getUniformRandomValueGenerator(double lowBoundary,
                                         double highBoundary)
{
    return [=] () -> double
    {
        return lowBoundary + (highBoundary - lowBoundary) * rand() / RAND_MAX;
    };
}

double monteCarloIntegration(double lowBoundary,
                             double highBoundary,
                             int numberOfApproximations,
                             lambda integratedFunction,
                             lambda distributionDensityFunction,
                             generator randomValueGenerator)
{
    int size;
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    srand(time(NULL));

    double integrationResult                = 0.;
    int    numberOfApproximationsForProcess = numberOfApproximations / size;

    int currentNumber = numberOfApproximationsForProcess;
    if (rank < numberOfApproximations % size)
    {
        numberOfApproximationsForProcess++;
    }

    double localSum = 0.;

    for (int i = 0; i < numberOfApproximationsForProcess; i++)
    {
        double x = randomValueGenerator();

        localSum += integratedFunction(x) / distributionDensityFunction(x);
    }

    MPI_Reduce(&localSum, &integrationResult, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    return integrationResult / numberOfApproximations;
}

double monteCarloIntegration(double lowBoundary,
                             double highBoundary,
                             int numberOfApproximations,
                             lambda integratedFunction)
{
    return monteCarloIntegration(lowBoundary,
                                 highBoundary,
                                 numberOfApproximations,
                                 integratedFunction,
                                 getUniformDistributionDensity(lowBoundary, highBoundary),
                                 getUniformRandomValueGenerator(lowBoundary,highBoundary));
}
