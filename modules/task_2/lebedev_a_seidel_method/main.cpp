// Copyright 2019 Lebedev Alexander
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include "./seidel_method.h"

TEST(Seqential_Seidel, Test_Light_Matrix) {
    int n = 3;
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    matrix a;
    std::vector<double> b;

    if (rank == 0) {
        a.init(n, n);
        b.resize(n);

        a[0][0] = 10; a[0][1] = 1; a[0][2] = 1;
        a[1][0] = 2; a[1][1] = 10; a[1][2] = 1;
        a[2][0] = 2; a[2][1] = 2; a[2][2] = 10;

        b[0] = 12; b[1] = 13; b[2] = 14;
    }
    double eps = 0.00001;
    bool bad = false;

    std::vector<double> expectedVector(n, 1.);
    std::vector<double> result;
    if (rank == 0) {
        result = solveSequentialSeidel(a, b, eps);
    }
    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            if (std::fabs(expectedVector[i] - result[i]) >= eps) bad = true;
        }
    }
    EXPECT_EQ(bad, false);
}

TEST(Parallel_Seidel, Test_Light_Matrix) {
    int n = 3;
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    matrix a;
    std::vector<double> b;

    if (rank == 0) {
        a.init(n, n);
        b.resize(n);

        a[0][0] = 10; a[0][1] = 1; a[0][2] = 1;
        a[1][0] = 2; a[1][1] = 10; a[1][2] = 1;
        a[2][0] = 2; a[2][1] = 2; a[2][2] = 10;

        b[0] = 12; b[1] = 13; b[2] = 14;
    }
    double eps = 0.00001;
    bool bad = false;

    std::vector<double> expectedVector(n, 1.);
    auto result = solveParallelSeidel(a, b, eps);
    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            if (std::fabs(expectedVector[i] - result[i]) >= eps) bad = true;
        }
    }
    EXPECT_EQ(bad, false);
}

TEST(Parallel_Seidel_MPI, Test_check_Parallel_and_Seq_on_Rand_3) {
    int n = 3;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    matrix a;
    std::vector<double> b;

    if (rank == 0) {
        a.init(getRandomMatrix(n));
        b = getRandomVector(n);
    }

    double eps = 0.00001;
    bool bad = false;

    std::vector<double> resultSequential;
    std::vector<double> resultParallel = solveParallelSeidel(a, b, eps);
    if (rank == 0) {
        resultSequential = solveSequentialSeidel(a, b, eps);
    }

    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            double l1 = static_cast<int>(resultParallel[i] * 1000000) / 1000000;
            double l2 = static_cast<int>(resultSequential[i] * 1000000) / 1000000;

            if (std::fabs(l1 - l2) > eps) {
                bad = true;
            }
        }
        EXPECT_EQ(bad, false);
    }
}

TEST(Parallel_Seidel_MPI, Test_check_Parallel_and_Seq_on_Rand_6) {
    int n = 6;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    matrix a;
    std::vector<double> b;

    if (rank == 0) {
        a.init(getRandomMatrix(n));
        b = getRandomVector(n);
    }

    double eps = 0.00001;
    bool bad = false;

    std::vector<double> resultSequential;
    std::vector<double> resultParallel = solveParallelSeidel(a, b, eps);
    if (rank == 0) {
        resultSequential = solveSequentialSeidel(a, b, eps);
    }

    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            double l1 = static_cast<int>(resultParallel[i] * 1000000) / 1000000;
            double l2 = static_cast<int>(resultSequential[i] * 1000000) / 1000000;

            if (std::fabs(l1 - l2) > eps) {
                bad = true;
            }
        }
        EXPECT_EQ(bad, false);
    }
}

TEST(Parallel_Seidel_MPI, Test_check_Parallel_and_Seq_on_Rand_10) {
    int n = 10;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    matrix a;
    std::vector<double> b;

    if (rank == 0) {
        a.init(getRandomMatrix(n));
        b = getRandomVector(n);
    }

    double eps = 0.00001;
    bool bad = false;

    std::vector<double> resultSequential;
    std::vector<double> resultParallel = solveParallelSeidel(a, b, eps);
    if (rank == 0) {
        resultSequential = solveSequentialSeidel(a, b, eps);
    }

    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            double l1 = static_cast<int>(resultParallel[i] * 1000000) / 1000000;
            double l2 = static_cast<int>(resultSequential[i] * 1000000) / 1000000;

            if (std::fabs(l1 - l2) > eps) {
                bad = true;
            }
        }
        EXPECT_EQ(bad, false);
    }
}

TEST(Parallel_Seidel_MPI, Test_check_Parallel_and_Seq_on_Rand_1000) {
    int n = 1000;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    matrix a;
    std::vector<double> b;

    if (rank == 0) {
        a.init(getRandomMatrix(n));
        b = getRandomVector(n);
    }

    double eps = 0.00001;
    bool bad = false;

    std::vector<double> resultSequential;

    double sequentialTime = 0.;

    double parallelTime = MPI_Wtime(); 
    std::vector<double> resultParallel = solveParallelSeidel(a, b, eps);
    parallelTime = MPI_Wtime() - parallelTime;

    if (rank == 0) {
        sequentialTime = MPI_Wtime();
        resultSequential = solveSequentialSeidel(a, b, eps);
        sequentialTime = MPI_Wtime() - sequentialTime;
    }

    if (rank == 0) {
        std::cerr << "Sequantial Time: " << sequentialTime << " | ParallelTime: " << parallelTime << "\n";
        for (int i = 0; i < n; i++) {
            double l1 = static_cast<int>(resultParallel[i] * 1000000) / 1000000;
            double l2 = static_cast<int>(resultSequential[i] * 1000000) / 1000000;

            if (std::fabs(l1 - l2) > eps) {
                bad = true;
            }
        }
        EXPECT_EQ(bad, false);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners &listeners =
            ::testing::UnitTest::GetInstance()->listeners();

    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());

    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);

    return RUN_ALL_TESTS();
}
