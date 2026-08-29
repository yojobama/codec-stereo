cmake_minimum_required(VERSION 3.16)
project(codec_stereo C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CS_ENABLE_REF_SAD "Build the brute-force SAD diagnostic backend (CS_MODE_DIRECT)" ON)
option(CS_ENABLE_LAVC    "Build the libavcodec/libx264 encode+decode reference backend" OFF)
option(CS_ENABLE_RKMPP   "Build the Rockchip MPP encode+readback backend (rk3588)" OFF)
option(CS_ENABLE_D3D12   "Build the D3D12 Video Motion Estimation backend (Windows)" OFF)
option(CS_BUILD_TOOLS    "Build CLI tools (cs_depth, cs_bench)" ON)
option(CS_BUILD_TESTS    "Build unit tests" ON)
option(CS_BUILD_HARNESS  "Build the OpenCV validation harness (needs OpenCV)" OFF)

add_library(codec_stereo
    src/cs_core.c
    src/cs_disparity.c
    src/cs_image.c
)
target_include_directories(codec_stereo
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(CS_ENABLE_REF_SAD)
    target_sources(codec_stereo PRIVATE src/backends/cs_backend_ref_sad.c)
    target_compile_definitions(codec_stereo PUBLIC CS_ENABLE_REF_SAD)
endif()

if(CS_ENABLE_LAVC)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET libavcodec libavutil libavformat)
    target_sources(codec_stereo PRIVATE src/backends/cs_backend_lavc_sw.c)
    target_compile_definitions(codec_stereo PUBLIC CS_ENABLE_LAVC)
    target_link_libraries(codec_stereo PUBLIC PkgConfig::LIBAV)
endif()

if(CS_ENABLE_RKMPP)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(ROCKCHIP_MPP QUIET rockchip_mpp)
    if(NOT ROCKCHIP_MPP_FOUND)
        find_library(ROCKCHIP_MPP_LIB rockchip_mpp REQUIRED)
        set(ROCKCHIP_MPP_LIBRARIES ${ROCKCHIP_MPP_LIB})
    endif()
    target_sources(codec_stereo PRIVATE src/backends/cs_backend_rkmpp.c)
    target_compile_definitions(codec_stereo PUBLIC CS_ENABLE_RKMPP)
    target_link_libraries(codec_stereo PUBLIC ${ROCKCHIP_MPP_LIBRARIES})
    if(ROCKCHIP_MPP_INCLUDE_DIRS)
        target_include_directories(codec_stereo PRIVATE ${ROCKCHIP_MPP_INCLUDE_DIRS})
    endif()
endif()

if(CS_ENABLE_D3D12)
    enable_language(CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    target_sources(codec_stereo PRIVATE src/backends/cs_backend_d3d12_vme.cpp)
    target_compile_definitions(codec_stereo PUBLIC CS_ENABLE_D3D12)
    target_link_libraries(codec_stereo PUBLIC d3d12 dxgi)
endif()

if(CS_BUILD_TOOLS)
    add_executable(cs_depth tools/cs_depth.c)
    target_link_libraries(cs_depth PRIVATE codec_stereo)

    add_executable(cs_bench tools/cs_bench.c)
    target_link_libraries(cs_bench PRIVATE codec_stereo)
endif()

if(CS_BUILD_TESTS)
    enable_testing()
    add_executable(test_calibration tests/test_calibration.c)
    target_link_libraries(test_calibration PRIVATE codec_stereo)
    add_test(NAME calibration COMMAND test_calibration)
endif()

if(CS_BUILD_HARNESS)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d)
    add_executable(cs_eval harness/cs_eval.cpp)
    target_link_libraries(cs_eval PRIVATE codec_stereo ${OpenCV_LIBS})
    target_include_directories(cs_eval PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()
