// Copyright 2019 Gribanov Mihail
#include "../../../modules/task_2/gribanov_m_image_smoothing/image_smoothing.h"
#include <mpi.h>
#include <vector>
#include <random>
#include <ctime>

std::vector<int> getImg(int sizeX, int sizeY) {
    if (sizeX <= 0 || sizeY <= 0) {
        throw - 1;
    }
    std::vector <int> arr(sizeX * sizeY);
    std::default_random_engine r;
    r.seed(static_cast<unsigned int>(std::time(0)));
    for (int i = 0; i < sizeX * sizeY; i++) {
        arr[i] = r() % 256;
    }
    return arr;
}

std::vector<int> smoothSeq(std::vector<int> img, int sizeX, int sizeY) {
    std::vector<int> newImg(sizeX * sizeY);

    for (int i = 0; i < sizeX*sizeY; i++) {
        int sum = img[i];
        int n = 1;

        if (i % sizeX) {
            sum += img[i - 1];
            n++;
        }
        if ((i + 1) % sizeX) {
            sum += img[i + 1];
            n++;
        }
        if (i > sizeX) {
            sum += img[i - sizeX];
            n++;
            if (i % sizeX) {
                sum += img[i - sizeX - 1];
                n++;
            }
            if ((i + 1) % sizeX) {
                sum += img[i - sizeX + 1];
                n++;
            }
        }
        if (i / sizeX + 1 < sizeY) {
            sum += img[i + sizeX];
            n++;
            if (i % sizeX) {
                sum += img[i + sizeX - 1];
                n++;
            }
            if ((i + 1) % sizeX) {
                sum += img[i + sizeX + 1];
                n++;
            }
        }
        newImg[i] = sum / n;
    }
    return newImg;
}

std::vector<int> smoothPar(std::vector<int> img, int sizeX, int sizeY) {
    int size, rank;
    std::vector<int> newImg(sizeX * sizeY);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int delta = sizeX*sizeY / size;
    int rem = sizeX*sizeY % size;

    std::vector<int> startPos(size);
    for (int i = 1; i < size; i++)
        startPos[i] = i * delta + rem;

    if (rank == 0) {
        if (delta > 0) {
            for (int proc = 1; proc < size; proc++) {
                MPI_Send(&startPos[proc], 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        if (delta > 0) {
            int startPos = -1;
            MPI_Status status;
            MPI_Recv(&startPos, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            for (int i = startPos; i < startPos + delta; i++) {
                int sum = img[i];
                int n = 1;
                if (i % sizeX) {
                    sum += img[i - 1];
                    n++;
                }
                if ((i + 1) % sizeX) {
                    sum += img[i + 1];
                    n++;
                }
                if (i > sizeX) {
                    sum += img[i - sizeX];
                    n++;
                    if (i % sizeX) {
                        sum += img[i - sizeX - 1];
                        n++;
                    }
                    if ((i + 1) % sizeX) {
                        sum += img[i - sizeX + 1];
                        n++;
                    }
                }
                if (i / sizeX + 1 < sizeY) {
                    sum += img[i + sizeX];
                    n++;
                    if (i % sizeX) {
                        sum += img[i + sizeX - 1];
                        n++;
                    }
                    if ((i + 1) % sizeX) {
                        sum += img[i + sizeX + 1];
                        n++;
                    }
                }
                newImg[i] = sum / n;
            }
            MPI_Send(&newImg[startPos], delta, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
    if (rank == 0) {
        MPI_Status status;
        for (int i = 0; i < delta + rem; i++) {
            int sum = img[i];
            int n = 1;
            if (i % sizeX) {
                sum += img[i - 1];
                n++;
            }
            if ((i + 1) % sizeX) {
                sum += img[i + 1];
                n++;
            }
            if (i > sizeX) {
                sum += img[i - sizeX];
                n++;
                if (i % sizeX) {
                    sum += img[i - sizeX - 1];
                    n++;
                }
                if ((i + 1) % sizeX) {
                    sum += img[i - sizeX + 1];
                    n++;
                }
            }
            if (i / sizeX + 1 < sizeY) {
                sum += img[i + sizeX];
                n++;
                if (i % sizeX) {
                    sum += img[i + sizeX - 1];
                    n++;
                }
                if ((i + 1) % sizeX) {
                    sum += img[i + sizeX + 1];
                    n++;
                }
            }
            newImg[i] = sum / n;
        }
        for (int j = 1; j < size; j++) {
            MPI_Recv(&newImg[startPos[j]], delta, MPI_INT, j, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        }
    }
    return newImg;
}
