// Copyright 2019 Lebedev Alexander
#ifndef MODULES_TASK_2_LEBEDEV_A_SEIDEL_METHOD_SEIDEL_METHOD_H_
#define MODULES_TASK_2_LEBEDEV_A_SEIDEL_METHOD_SEIDEL_METHOD_H_

#include <vector>
#include <utility>
#include <cmath>

void transpanent(std::vector<double> &v, int n);

std::vector<double> getRandomVector(int n, int lowerBound = 0, int upperBound = 10);

std::vector<double> getRandomMatrix(int n, int lowerBound = 0, int upperBound = 10);

std::vector<double> solveSequentialSeidel(std::vector<double> a,
                                          std::vector<double> b,
                                          double eps = 0.0000001,
                                          int approximationsCount = 1000);


std::vector<double> solveParallelSeidel(std::vector<double> a,
                                        std::vector<double> b,
                                        double eps = 0.0000001,
                                        int approximationsCount = 1000,
                                        MPI_Comm comm = MPI_COMM_WORLD);

#endif  // MODULES_TASK_2_LEBEDEV_A_SEIDEL_METHOD_SEIDEL_METHOD_H_
