cmake_minimum_required(VERSION 3.10)

project(CascadeClassifier)

add_subdirectory(traincascade)

add_subdirectory(tools/annotation)

add_subdirectory(tools/createsamples)

add_subdirectory(tools/detection/Cpp)

add_subdirectory(tools/visualisation)

