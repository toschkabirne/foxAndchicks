# Instructions

## Prerequisites

 - CMake https://cmake.org/download/

## Build

1. Go inside the source directory
```
cd predator-vs-prey
```
2. In a terminal:

If you are using the **MSVC** compiler:
```
mkdir build
cd build
cmake ..
cmake --build . -t full_version --config Release
```
else
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. 
cmake --build . -t full_version
```