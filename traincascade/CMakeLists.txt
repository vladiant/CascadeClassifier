cmake_minimum_required(VERSION 3.10)

project(traincascade LANGUAGES CXX)

find_package(OpenCV REQUIRED)

add_executable(
    ${PROJECT_NAME}
    o_cvdtreeparams.h
    o_cvdtreeparams.cpp
    o_cvboostparams.h
    o_cvboostparams.cpp
    o_cvstatmodel.h
    o_cvstatmodel.cpp
    o_cvboost.h
    o_cvboost.cpp
    o_cvdtreesplit.h
    o_cvdtreenode.h
    o_cvdtreetraindata.h
    o_cvdtreetraindata.cpp
    o_blockedrange.h
    o_cvdtree.h
    o_cvdtree.cpp
    o_cvboostree.h
    o_cvboostree.cpp
    boost.h
    boost.cpp
    cascadeclassifier.h
    cascadeclassifier.cpp
    features.cpp
    haarfeatures.h
    haarfeatures.cpp
    HOGfeatures.h
    HOGfeatures.cpp
    imagestorage.h
    imagestorage.cpp
    lbpfeatures.h
    lbpfeatures.cpp
    traincascade_features.h
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
        ${OpenCV_LIBS}
)

