// Copyright 2019 Lebedev Alexander
#ifndef MODULES_TASK_2_LEBEDEV_A_SEIDEL_METHOD_SEIDEL_METHOD_H_
#define MODULES_TASK_2_LEBEDEV_A_SEIDEL_METHOD_SEIDEL_METHOD_H_

#include <vector>
#include <exception>
#include <utility>
#include <cmath>

struct matrix {
    std::vector<double> v;
    int n;
    int m;

    matrix(int newN, int newM) {
        n = newN;
        m = newM;
        v = std::vector<double>(n * m);
    }

    explicit matrix(int newN = 0) {
        n = m = newN;
        v = std::vector<double>(n * m);
    }

    void init(int newN, int newM) {
        n = newN;
        m = newM;
        v = std::vector<double>(n * m);
    }

    void init(const matrix &x) {
        n = x.n;
        m = x.m;
        v = x.v;
    }

    void transpanent() {
        if (n != m) throw std::exception("Matrix to transpanent is not sqare");
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < m; j++) {
                std::swap(v[i * m + j], v[j * m + i]);
            }
        }
    }

    auto operator[](int x) {
        if (x >= n) throw std::out_of_range("Matrix: i >= n");
        return &v[x * m];
    }
};

std::vector<double> getRandomVector(int n, int lowerBound = 0, int upperBound = 10);

matrix getRandomMatrix(int n, int lowerBound = 0, int upperBound = 10);

std::vector<double> solveSequentialSeidel(matrix a,
                                          std::vector<double> b,
                                          double eps = 0.0000001,
                                          int approximationsCount = 1000);


std::vector<double> solveParallelSeidel(matrix a,
                                        std::vector<double> b,
                                        double eps = 0.0000001,
                                        int approximationsCount = 1000,
                                        MPI_Comm comm = MPI_COMM_WORLD);

#endif  // MODULES_TASK_2_LEBEDEV_A_SEIDEL_METHOD_SEIDEL_METHOD_H_
