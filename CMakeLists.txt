cmake_minimum_required(VERSION 3.10)

project(CascadeClassifier)

# Add Opencv for Windows and MacOS
if (WIN32 OR APPLE)
    include(ExternalProject)

    set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

    ExternalProject_Add(opencv
        GIT_REPOSITORY https://github.com/opencv/opencv
        GIT_TAG 4.2.0
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} -DBUILD_DOCS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DBUILD_opencv_apps=OFF
    )

    include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
    link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${EXTERNAL_INSTALL_LOCATION}/lib/cmake/opencv4/)
    message(status "Running external OpenCV build")
endif()

add_subdirectory(traincascade)

add_subdirectory(tools/annotation)

add_subdirectory(tools/createsamples)

add_subdirectory(tools/detection/Cpp)

add_subdirectory(tools/visualisation)

