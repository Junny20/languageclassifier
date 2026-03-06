cmake_minimum_required(VERSION 3.18)

project(LanguageClassifier CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# uses CMAKE_PREFIX_PATH built-in variable to find LibTorch package
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(train 
    src/train.cpp
    src/dataset.cpp
    src/tokenizer.cpp
    src/model.cpp
)

target_link_libraries(train ${TORCH_LIBRARIES})

target_include_directories(train PRIVATE src/ third_party/)