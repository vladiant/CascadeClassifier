cmake_minimum_required(VERSION 3.10)

project(CascadeDetect)

find_package(OpenCV)

add_executable(
    ${PROJECT_NAME}
    main.cpp
)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

if (WIN32 OR APPLE)
    add_dependencies(${PROJECT_NAME} opencv)
endif()

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
        ${OpenCV_LIBS}
)
