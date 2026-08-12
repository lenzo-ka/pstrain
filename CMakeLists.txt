cmake_minimum_required(VERSION 3.25)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" PSTRAIN_PYPROJECT)
string(REGEX MATCH "version[ \t]*=[ \t]*\"([^\"]+)\"" _ "${PSTRAIN_PYPROJECT}")
if(NOT CMAKE_MATCH_1)
  message(FATAL_ERROR "Could not read project.version from pyproject.toml")
endif()

project(pstrain VERSION ${CMAKE_MATCH_1}
  DESCRIPTION "pstrain, the Peace Train - acoustic model training toolkit"
  HOMEPAGE_URL "https://github.com/lenzo-ka/pstrain"
  LANGUAGES C)

include(CMakePrintHelpers)
include(CheckTypeSize)
include(CheckSymbolExists)
include(CheckIncludeFile)
include(TestBigEndian)
include(GNUInstallDirs)
include(FetchContent)

# Package info
set(PACKAGE_NAME ${PROJECT_NAME})
set(PACKAGE_VERSION ${PROJECT_VERSION})
set(PACKAGE_STRING "${PROJECT_NAME} ${PROJECT_VERSION}")

# Platform checks
test_big_endian(WORDS_BIGENDIAN)
CHECK_TYPE_SIZE(long LONG)
CHECK_TYPE_SIZE("long long" LONG_LONG)
set(SIZEOF_LONG ${LONG})
set(SIZEOF_LONG_LONG ${LONG_LONG})
CHECK_SYMBOL_EXISTS(popen stdio.h HAVE_POPEN)
CHECK_SYMBOL_EXISTS(snprintf stdio.h HAVE_SNPRINTF)
CHECK_INCLUDE_FILE(sys/stat.h HAVE_SYS_STAT_H)
CHECK_INCLUDE_FILE(sys/types.h HAVE_SYS_TYPES_H)
CHECK_INCLUDE_FILE(unistd.h HAVE_UNISTD_H)
CHECK_INCLUDE_FILE(inttypes.h HAVE_INTTYPES_H)
CHECK_INCLUDE_FILE(stdint.h HAVE_STDINT_H)
CHECK_INCLUDE_FILE(errno.h HAVE_ERRNO_H)

# Compiler flags
if(MSVC)
  add_compile_options(/W3)
else()
  add_compile_options(-Wall -Wextra)
  add_compile_options(
    -Wno-sign-compare
    -Wno-unused-parameter
    -Wno-unused-but-set-variable
    -Wno-pointer-sign
    -Wno-missing-field-initializers
  )
endif()

# Build options
option(BUILD_SHARED_LIBS "Build shared library (required for Python bindings)" ON)
option(BUILD_CLI "Build CLI programs for development/testing" ON)

# PocketSphinx is a source dependency, pinned by both immutable commit and
# archive digest.  Keep all integration changes here: the upstream source is
# built verbatim.
set(PSTRAIN_POCKETSPHINX_VERSION "5.1.1" CACHE INTERNAL
    "PocketSphinx version included in the pstrain measurement identity")
set(PSTRAIN_POCKETSPHINX_COMMIT
    "511126b492dcb267cf30d49d631946d7b61a9530" CACHE INTERNAL
    "PocketSphinx source commit")
FetchContent_Declare(pocketsphinx_source
  URL "https://github.com/cmusphinx/pocketsphinx/archive/${PSTRAIN_POCKETSPHINX_COMMIT}.tar.gz"
  URL_HASH "SHA256=9e20444c67c3381a0cef68a125c9c6b2830fdac278c5344dd6eab17228c988aa"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(pocketsphinx_source)

# PocketSphinx 5.1.1 supports FetchContent but uses top-level source/binary
# include paths.  Add the subproject paths at our call boundary without
# changing any upstream files.
target_include_directories(pocketsphinx PUBLIC
  "${pocketsphinx_source_SOURCE_DIR}/include"
  "${pocketsphinx_source_BINARY_DIR}/include"
)
set_target_properties(pocketsphinx PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
if(UNIX AND NOT APPLE)
  # Prevent ELF symbol interposition with the intentionally vendored older
  # sphinxbase implementation exported by libpstrainc.
  target_link_options(pocketsphinx PRIVATE "LINKER:-Bsymbolic")
endif()

# Output directories (for development builds)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Configure header
configure_file(csrc/config.h.in config.h)
add_definitions(-DHAVE_CONFIG_H)

# Detect scikit-build-core wheel build
if(SKBUILD)
  message(STATUS "Building Python wheel via scikit-build-core")
  # For wheel builds, install to the wheel root (pstrain/_lib/)
  # scikit-build-core sets wheel.install-dir in pyproject.toml
  set(PSTRAIN_INSTALL_LIBDIR "lib")
  set(PSTRAIN_INSTALL_BINDIR "bin")
  set(PSTRAIN_INSTALL_INCLUDEDIR "include")
  # All programs go to bin/ in wheel
  set(PSTRAIN_INSTALL_PROGRAMS "bin")
else()
  # Standard system install
  set(PSTRAIN_INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR})
  set(PSTRAIN_INSTALL_BINDIR ${CMAKE_INSTALL_BINDIR})
  set(PSTRAIN_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}/pstrain)
  # Internal programs go to libexec/pstrainc/
  set(PSTRAIN_INSTALL_PROGRAMS ${CMAKE_INSTALL_LIBEXECDIR}/pstrainc)
endif()

cmake_print_variables(SKBUILD PSTRAIN_INSTALL_LIBDIR PSTRAIN_INSTALL_BINDIR PSTRAIN_INSTALL_PROGRAMS)

# Enable CTest at the top level so `ctest --test-dir <build>` recurses into
# the csrc/tests registered below. Must be called before add_subdirectory.
enable_testing()

# Build the library
add_subdirectory(csrc)
