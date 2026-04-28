cmake_minimum_required(VERSION 3.10)

project(traincascade LANGUAGES CXX)

find_package(OpenCV REQUIRED)

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
        ${OpenCV_INCLUDE_DIRS}
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
         $<$<CXX_COMPILER_ID:MSVC>:/Zc:__cplusplus /Zc:preprocessor /Zc:inline>
         $<$<CXX_COMPILER_ID:MSVC>:/w14191 /w14242 /w14254 /w14263 /w14265 /w14266 /w14287>
         $<$<CXX_COMPILER_ID:MSVC>:/w14289 /w14296 /w14311 /w14545 /w14546 /w14547 /w14549>
         $<$<CXX_COMPILER_ID:MSVC>:/w14555 /w14619 /w14640 /w14826 /w14905 /w14906 /w14928>
        #  $<$<CXX_COMPILER_ID:MSVC>:/w14365> # Fix in code
         $<$<CXX_COMPILER_ID:MSVC>:/w14062 /w14061 /w15262 /w14463 /w14146>
         $<$<CXX_COMPILER_ID:MSVC>:/w14456 /w14457 /w14458 /w14459>
         $<$<CXX_COMPILER_ID:MSVC>:/w14700 /w14701 /w14703 /w14774 /w14777>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
        #  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wold-style-cast> # Fix in code
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wcast-qual>
        #  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wsign-conversion> # Fix in code
        #  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wconversion> # Fix in code
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wshadow -Wnull-dereference -Wmisleading-indentation>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wimplicit-fallthrough -Wswitch-enum>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wshift-negative-value>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wformat=2 -Wformat-security>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fstrict-aliasing>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wstack-protector -fstack-protector-strong>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wnon-virtual-dtor -Woverloaded-virtual>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wuninitialized>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Walloca -Wvla>
         $<$<CXX_COMPILER_ID:GNU>:-Wcast-align=strict>
         $<$<CXX_COMPILER_ID:GNU>:-Warith-conversion>
         $<$<CXX_COMPILER_ID:GNU>:-Wduplicated-branches>
         $<$<CXX_COMPILER_ID:GNU>:-Wduplicated-cond>
         $<$<CXX_COMPILER_ID:GNU>:-Wlogical-op>
         $<$<CXX_COMPILER_ID:GNU>:-Wshift-overflow=2>
         $<$<CXX_COMPILER_ID:GNU>:-Wstrict-aliasing=3>
         $<$<CXX_COMPILER_ID:GNU>:-Wmaybe-uninitialized>
         $<$<CXX_COMPILER_ID:Clang>:-Wcast-align>
         $<$<CXX_COMPILER_ID:Clang>:-Wlogical-op-parentheses>
         $<$<CXX_COMPILER_ID:Clang>:-Wshift-overflow>
         $<$<CXX_COMPILER_ID:Clang>:-Wconditional-uninitialized -Wsometimes-uninitialized>
)

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
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
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    test_traincascade
    PRIVATE
        TrainCascadeLib
)

add_test(NAME test_traincascade COMMAND test_traincascade)
