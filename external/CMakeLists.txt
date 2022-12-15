cmake_minimum_required(VERSION 3.10)

project(opencv)

include(ExternalProject)

set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

ExternalProject_Add(opencv
    GIT_REPOSITORY https://github.com/opencv/opencv.git
    GIT_TAG 4.5.0
    INSTALL_DIR ${EXTERNAL_INSTALL_LOCATION}
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
        -DBUILD_DOCS=OFF
        -DBUILD_EXAMPLES=OFF
        -DBUILD_PERF_TESTS=OFF
        -DBUILD_TESTS=OFF
        -DBUILD_opencv_apps=OFF
        -DBUILD_LIST=core,highgui,imgcodecs,imgproc,ml,dnn,videoio,calib3d,objdetect
        -DBUILD_opencv_python2=OFF
        -DBUILD_opencv_python3=OFF
        -DBUILD_PROTOBUF=OFF
        -DBUILD_opencv_dnn=OFF
)
