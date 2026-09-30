# Defines the INTERFACE target cs_ffmpeg (libavcodec + libavutil).
#
# Tries pkg-config first (Linux, and vcpkg builds where pkgconf is present),
# then falls back to find_path/find_library, which works for vcpkg under both
# MSVC and MinGW without needing pkg-config installed.
if(TARGET cs_ffmpeg)
    return()
endif()

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(CS_LIBAV QUIET IMPORTED_TARGET libavcodec libavutil)
endif()

add_library(cs_ffmpeg INTERFACE)
if(CS_LIBAV_FOUND)
    target_link_libraries(cs_ffmpeg INTERFACE PkgConfig::CS_LIBAV)
else()
    find_path(CS_AVCODEC_INCLUDE_DIR libavcodec/avcodec.h REQUIRED)
    find_library(CS_AVCODEC_LIB NAMES avcodec REQUIRED)
    find_library(CS_AVUTIL_LIB NAMES avutil REQUIRED)
    target_include_directories(cs_ffmpeg INTERFACE ${CS_AVCODEC_INCLUDE_DIR})
    target_link_libraries(cs_ffmpeg INTERFACE ${CS_AVCODEC_LIB} ${CS_AVUTIL_LIB})
endif()
