cmake_minimum_required(VERSION 3.10)

project(opencv)

# Add Opencv for Windows and MacOS
if (WIN32 OR APPLE)
    include(ExternalProject)

    set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

    ExternalProject_Add(opencv
        GIT_REPOSITORY https://github.com/opencv/opencv.git
        GIT_TAG 4.2.0
        INSTALL_DIR ${EXTERNAL_INSTALL_LOCATION}
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} -DBUILD_DOCS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DBUILD_opencv_apps=OFF
    )

    message(STATUS "Running external OpenCV build")

    list(APPEND CMAKE_MODULE_PATH "${EXTERNAL_INSTALL_LOCATION}/lib/cmake/opencv4/")OpenCV_DIR
    set(ENV{CMAKE_MODULE_PATH} "${CMAKE_MODULE_PATH}")
endif()
