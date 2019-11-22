// Copyright 2019 Gribanov Mihail

#ifndef MODULES_TASK_2_GRIBANOV_M_IMAGE_SMOOTHING_IMAGE_SMOOTHING_H_
#define MODULES_TASK_2_GRIBANOV_M_IMAGE_SMOOTHING_IMAGE_SMOOTHING_H_

#include <vector>

std::vector<int> getImg(int sizeX, int sizeY);
std::vector<int> smoothSeq(std::vector<int> img, int sizeX, int sizeY);
std::vector<int> smoothPar(std::vector<int> img, int sizeX, int sizeY);

#endif  // MODULES_TASK_2_GRIBANOV_M_IMAGE_SMOOTHING_IMAGE_SMOOTHING_H_
