// Copyright 2019 Lebedev Alexander

#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <math.h>
#include <vector>
#include "../../../modules/task_1/lebedev_a_monte_carlo/monte_carlo.h"

TEST(monte_carlo_MPI, test1) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    lambda integratedFunction = [] (double x) -> double {
                return cos(x) * tanh(x) / (x * x);
            };

    double expectedValue = 0.0177504;
    double eps           = 0.08;

    double receivedValue = monteCarloIntegration(1., 2., 1000000, integratedFunction);

    if (rank == 0) {
        ASSERT_NEAR(expectedValue, receivedValue, eps);
    }
}

TEST(monte_carlo_MPI, test2) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    lambda integratedFunction = [] (double x) -> double {
        return cos(x);
    };

    double expectedValue = 1.;
    double eps           = 0.005;
    double pi            = acos(-1.);

    double receivedValue = monteCarloIntegration(0., pi / 2, 1000000, integratedFunction);

    if (rank == 0) {
        ASSERT_NEAR(expectedValue, receivedValue, eps);
    }
}

TEST(monte_carlo_MPI, test3) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    lambda integratedFunction = [] (double x) -> double {
        return x * x * x * 10 + x * x + 2.4;
    };

    double expectedValue = 14.093331476;
    double eps           = 0.05;

    double receivedValue = monteCarloIntegration(-1.3, 1.5, 10000000, integratedFunction);

    if (rank == 0) {
        ASSERT_NEAR(expectedValue, receivedValue, eps);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners& listeners =
            ::testing::UnitTest::GetInstance()->listeners();

    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());

    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
    return RUN_ALL_TESTS();
}
