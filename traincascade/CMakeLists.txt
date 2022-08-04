cmake_minimum_required(VERSION 3.10)

project(traincascade LANGUAGES CXX)

find_package(OpenCV)

# Library
add_library(
    TrainCascadeLib
    lib/include/o_utils.h
    lib/src/o_utils.cpp
    lib/include/o_cvdtreeparams.h
    lib/src/o_cvdtreeparams.cpp
    lib/include/o_cvboostparams.h
    lib/src/o_cvboostparams.cpp
    lib/include/o_cvstatmodel.h
    lib/src/o_cvstatmodel.cpp
    lib/include/o_cvboost.h
    lib/src/o_cvboost.cpp
    lib/include/o_cvdtreesplit.h
    lib/include/o_cvdtreenode.h
    lib/include/o_cvdtreetraindata.h
    lib/src/o_cvdtreetraindata.cpp
    lib/include/o_blockedrange.h
    lib/include/o_cvdtree.h
    lib/src/o_cvdtree.cpp
    lib/include/o_cvboostree.h
    lib/src/o_cvboostree.cpp
    lib/include/o_cvcascadeboosttraindata.h
    lib/src/o_cvcascadeboosttraindata.cpp
    lib/include/o_cvcascadeboosttree.h
    lib/src/o_cvcascadeboosttree.cpp
    lib/include/boost.h
    lib/src/boost.cpp
    lib/include/cascadeclassifier.h
    lib/src/cascadeclassifier.cpp
    lib/src/features.cpp
    lib/include/haarfeatures.h
    lib/src/haarfeatures.cpp
    lib/include/HOGfeatures.h
    lib/src/HOGfeatures.cpp
    lib/include/imagestorage.h
    lib/src/imagestorage.cpp
    lib/include/lbpfeatures.h
    lib/src/lbpfeatures.cpp
    lib/include/traincascade_features.h
)

set_target_properties(
    TrainCascadeLib
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_compile_options(
    TrainCascadeLib
    PRIVATE
         $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
)

target_include_directories(
    TrainCascadeLib
    PUBLIC
        lib/include/
)

target_link_libraries(
    TrainCascadeLib
    PUBLIC
        ${OpenCV_LIBS}
)

# Executable
add_executable(
    ${PROJECT_NAME}
    traincascade.cpp
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

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
        TrainCascadeLib
)

# Tests
include(CTest)

add_executable(
    test_traincascade
    test/main.cpp
)

set_target_properties(
    test_traincascade
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_compile_options(
    test_traincascade
    PRIVATE
         $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
)

target_include_directories(
    test_traincascade
    PRIVATE
        test
)

target_link_libraries(
    test_traincascade
    PRIVATE
        TrainCascadeLib
)

add_test(NAME test_traincascade COMMAND test_traincascade)
