cmake_minimum_required(VERSION 3.18)

project(LanguageClassifier CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ORT_DIR ${CMAKE_SOURCE_DIR}/third_party/onnxruntime-osx-arm64-1.20.1)

# uses CMAKE_PREFIX_PATH built-in variable to find LibTorch package
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(train
    src/train.cpp
    src/tokenizer.cpp
    src/dataset.cpp
    src/model.cpp
)

add_executable(server
    src/server.cpp
    src/tokenizer.cpp
)

target_link_libraries(train ${TORCH_LIBRARIES})
target_include_directories(train PRIVATE src/ third_party/ /opt/homebrew/include)

target_link_libraries(server ${TORCH_LIBRARIES} ${ORT_DIR}/lib/libonnxruntime.dylib)
target_include_directories(server PRIVATE src/ third_party/ /opt/homebrew/include ${ORT_DIR}/include)