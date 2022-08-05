cmake_minimum_required(VERSION 3.10)

project(createsamples LANGUAGES CXX)

find_package(OpenCV)
include(${EXTERNAL_INSTALL_LOCATION}/lib/cmake/opencv4/OpenCVConfig.cmake)

add_executable(
    ${PROJECT_NAME}
    createsamples.cpp
    utility.hpp
    utility.cpp
)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_compile_options(
    ${PROJECT_NAME}
    PRIVATE
         $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
)

if (WIN32 OR APPLE)
    add_dependencies(${PROJECT_NAME} opencv)
endif()

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
        ${OpenCV_LIBS}
)

