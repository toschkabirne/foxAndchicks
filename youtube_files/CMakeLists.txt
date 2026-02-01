cmake_minimum_required(VERSION 3.16)
project(Predator-vs-Prey LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

include(FetchContent)
FetchContent_Declare(SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 2.6.x)
FetchContent_MakeAvailable(SFML)

# Automatically adds all cpp files contained in the src directory
file(GLOB_RECURSE source_files
    "src/*.cpp"
)

set(SOURCES ${source_files})

function(create_default_target name)
   add_executable(${name} ${SOURCES})
   target_include_directories(${name} PRIVATE "src" "dep")
   target_link_libraries(${name} PRIVATE sfml-graphics)
   target_compile_features(${name} PRIVATE cxx_std_17)

   # Copy res dir to the binary directory
   add_custom_command(
       TARGET ${name}
       COMMENT "Copy Res directory"
       PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/res $<TARGET_FILE_DIR:${name}>/res
       VERBATIM)

   if(WIN32)
       add_custom_command(
           TARGET ${name}
           COMMENT "Copy OpenAL DLL"
           PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${SFML_SOURCE_DIR}/extlibs/bin/$<IF:$<EQUAL:${CMAKE_SIZEOF_VOID_P},8>,x64,x86>/openal32.dll $<TARGET_FILE_DIR:${name}>
           VERBATIM)
   endif()

   # Force static runtime linking
   if(MSVC)
       # Use MultiThreaded for Release and MultiThreadedDebug for Debug
       set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
   endif()
endfunction()

create_default_target(full_version)
create_default_target(demo)
target_compile_definitions(demo PRIVATE DEMO=1)

