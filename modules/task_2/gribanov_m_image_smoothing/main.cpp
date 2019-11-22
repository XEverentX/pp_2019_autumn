// Copyright 2019 Gribanov Mihail
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include "./image_smoothing.h"

TEST(Test_Image_Smoothing, Test_Get_Image) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        ASSERT_ANY_THROW(getImg(0, -1));
    }
}

TEST(Test_Image_Smoothing, Test_Sequential_Smooth) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> Img(9);
    std::vector<int> SImg(9);
    Img[0] = 96;
    Img[1] = 88;
    Img[2] = 174;
    Img[3] = 218;
    Img[4] = 134;
    Img[5] = 235;
    Img[6] = 254;
    Img[7] = 51;
    Img[8] = 1;
    SImg[0] = 134;
    SImg[1] = 157;
    SImg[2] = 157;
    SImg[3] = 164;
    SImg[4] = 139;
    SImg[5] = 113;
    SImg[6] = 164;
    SImg[7] = 148;
    SImg[8] = 105;
    Img = smoothSeq(Img, 3, 3);
    if (rank == 0) {
        ASSERT_EQ(Img, SImg);
    }
}

TEST(Test_Image_Smoothing, Test_Parallel_Smooth_3x3) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> Img(9);
    std::vector<int> SImg(9);
    Img = getImg(3, 3);
    SImg = smoothSeq(Img, 3, 3);
    Img = smoothPar(Img, 3, 3);
    if (rank == 0) {
        ASSERT_EQ(Img, SImg);
    }
}

TEST(Test_Image_Smoothing, Test_Parallel_Smooth_50x70) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> Img(50*70);
    std::vector<int> SImg(50*70);
    Img = getImg(50, 70);
    SImg = smoothSeq(Img, 50, 70);
    Img = smoothPar(Img, 50, 70);
    if (rank == 0) {
        ASSERT_EQ(Img, SImg);
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
