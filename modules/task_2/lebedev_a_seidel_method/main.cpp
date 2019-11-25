// Copyright 2019 Lebedev Alexander
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include "./seidel_method.h"

TEST(Seqential_Seidel, Test_Light_Matrix) {
    size_t n = 3;

    std::vector<std::vector<double>> a(n, std::vector<double>(n));
    a[0][0] = 10; a[0][1] = 1; a[0][2] = 1;
    a[1][0] = 2; a[1][1] = 10; a[1][2] = 1;
    a[2][0] = 2; a[2][1] = 2; a[2][2] = 10;

    std::vector<double> b(n);
    b[0] = 12; b[1] = 13; b[2] = 14;

    double eps = 0.00001;
    bool bad = false;

    std::vector<double> expectedVector(n, 1.);
    auto result = solveSequentialSeidel(a, b, eps);

    for (size_t i = 0; i < n; i++) {
        if (abs(expectedVector[i] - result[i]) >= eps) bad = true;
    }
    EXPECT_EQ(bad, false);
}

TEST(Parallel_Seidel_MPI, Test_Light_Matrix) {
    size_t n = 3;

    std::vector<std::vector<double>> a(n, std::vector<double>(n));
    a[0][0] = 10; a[0][1] = 1; a[0][2] = 1;
    a[1][0] = 2; a[1][1] = 10; a[1][2] = 1;
    a[2][0] = 2; a[2][1] = 2; a[2][2] = 10;

    std::vector<double> b(n);
    b[0] = 12; b[1] = 13; b[2] = 14;

    double eps = 0.00001;
    bool bad = false;

    std::vector<double> expectedVector(n, 1.);
    auto result = solveParallelSeidel(a, b, eps);

    for (size_t i = 0; i < n; i++) {
        if (abs(expectedVector[i] - result[i]) > eps) bad = true;
    }
    EXPECT_EQ(bad, false);
}

TEST(Parallel_Seidel_MPI, Test_check_Parallel_and_Seq_on_Rand_3) {
    size_t n = 3;

    auto a = getRandomMatrix(n);
    auto b = getRandomVector(n);

    double eps = 0.00001;
    bool bad = false;

    auto resultParallel = solveParallelSeidel(a, b, eps);
    auto resultSequential = solveSequentialSeidel(a, b, eps);

    for (size_t i = 0; i < n; i++) {
        double l1 = static_cast<int>(resultParallel[i] * 1000000) / 1000000;
        double l2 = static_cast<int>(resultSequential[i] * 1000000) / 1000000;

        if (abs(l1 - l2) > eps) {
            bad = true;
        }
    }
    EXPECT_EQ(bad, false);
}

TEST(Parallel_Seidel_MPI, Test_check_Parallel_and_Seq_on_Rand_6) {
    size_t n = 6;

    auto a = getRandomMatrix(n);
    auto b = getRandomVector(n);

    double eps = 0.00001;
    bool bad = false;

    auto resultParallel = solveParallelSeidel(a, b, eps);
    auto resultSequential = solveSequentialSeidel(a, b, eps);

    for (size_t i = 0; i < n; i++) {
        double l1 = static_cast<int>(resultParallel[i] * 1000000) / 1000000;
        double l2 = static_cast<int>(resultSequential[i] * 1000000) / 1000000;

        if (abs(l1 - l2) > eps) {
            bad = true;
        }
    }
    EXPECT_EQ(bad, false);
}

TEST(Parallel_Seidel_MPI, Test_check_Parallel_and_Seq_on_Rand_100) {
    size_t n = 100;

    auto a = getRandomMatrix(n);
    auto b = getRandomVector(n);

    double eps = 0.00001;
    bool bad = false;

    auto resultParallel = solveParallelSeidel(a, b, eps);
    auto resultSequential = solveSequentialSeidel(a, b, eps);

    for (size_t i = 0; i < n; i++) {
        double l1 = static_cast<int>(resultParallel[i] * 1000000) / 1000000;
        double l2 = static_cast<int>(resultSequential[i] * 1000000) / 1000000;

        if (abs(l1 - l2) > eps) {
            bad = true;
        }
    }
    EXPECT_EQ(bad, false);
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
