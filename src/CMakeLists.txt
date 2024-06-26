# Almost all CMake files should start with this
# You should always specify a range with the newest
# and oldest tested versions of CMake. This will ensure
# you pick up the best policies.
cmake_minimum_required(VERSION 3.1...3.29)

# This is your project statement. You should always list languages;
# Listing the version is nice here since it sets lots of useful variables
project(
  BlockedCompression
  VERSION 1.0
  LANGUAGES CXX)

# If you set any CMAKE_ variables, that can go here.
# (But usually don't do this, except maybe for C++ standard)
#target_compile_features(BlockedCompression PUBLIC cxx_std_20)
# Global set
set(CMAKE_CXX_STANDARD 20 CACHE STRING "Use C++ 20 standard")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Find packages go here.

# You should usually split this into folders, but this is a simple example

add_subdirectory(fmt)
add_subdirectory(c-blosc2)

# This is a "default" library, and will match the *** variable setting.
# Other common choices are STATIC, SHARED, and MODULE
# Including header files here helps IDEs but is not required.
# Output libname matches target name, with the usual extensions on your system
#add_library(MyLibExample simple_lib.cpp simple_lib.hpp)

# Link each target with other targets or add options, etc.

# Adding something we can run - Output name matches target name
add_executable(bc ./test.cpp)

target_include_directories(bc PUBLIC c-blosc2/include)
target_link_libraries(bc fmt::fmt blosc2_static)

# Make sure you link your targets with this command. It can also link libraries and
# even flags, so linking a target that does not exist will not give a configure-time error.
#target_link_libraries(MyExample PRIVATE MyLibExample)
