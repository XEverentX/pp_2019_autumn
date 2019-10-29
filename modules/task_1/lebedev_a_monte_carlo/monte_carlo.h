// Copyright 2019 Lebedev Alexander
#ifndef MODULES_TASK_1_LEBEDEV_A_MONTE_CARLO_H_
#define MODULES_TASK_1_LEBEDEV_A_MONTE_CARLO_H_

#include <vector>
#include <functional>

using lambda = std::function<double(double)>;
using generator = std::function<double()>;

lambda getUniformDistributionDensity(double lowBoundary,
                                     double highBoundary);

generator getUniformRandomValueGenerator(double lowBoundary,
                                         double highBoundary);

double monteCarloIntegration(double lowBoundary,
                             double highBoundary,
                             int numberOfApproximations,
                             lambda integratedFunction,
                             lambda distributionDensityFunction,
                             generator randomValueGenerator);

double monteCarloIntegration(double lowBoundary,
                             double highBoundary,
                             int numberOfApproximations,
                             lambda integratedFunction);

#endif  // MODULES_TASK_1_LEBEDEV_A_MONTE_CARLO_H_
