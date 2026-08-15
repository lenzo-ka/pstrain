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
include(CheckCCompilerFlag)
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
  # Keep the declared policy and compiler enforcement together, as in the
  # non-MSVC branch.  clang-cl accepts Clang flags through /clang:.
  set(PSTRAIN_FP_CONTRACT_POLICY "off" CACHE INTERNAL
      "Declared floating-point contraction policy for native targets")
  if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    set(PSTRAIN_FP_CONTRACT_FLAG "/clang:-ffp-contract=off")
  else()
    # Microsoft documents that /fp:precise defaults to fp_contract(off) only
    # starting with Visual Studio 2022 (MSVC 19.30, _MSC_VER 1930):
    # https://learn.microsoft.com/cpp/build/reference/fp-specify-floating-point-behavior
    if(MSVC_VERSION LESS 1930)
      message(FATAL_ERROR
        "MSVC ${MSVC_VERSION} is too old for the declared floating-point contraction "
        "policy: Visual Studio 2022 / MSVC 19.30 (_MSC_VER 1930) or newer is required")
    endif()
    set(PSTRAIN_FP_CONTRACT_FLAG "/fp:precise")
  endif()
  check_c_compiler_flag("${PSTRAIN_FP_CONTRACT_FLAG}"
    PSTRAIN_COMPILER_SUPPORTS_FP_CONTRACT)
  if(NOT PSTRAIN_COMPILER_SUPPORTS_FP_CONTRACT)
    message(FATAL_ERROR
      "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION} does not support the declared "
      "floating-point contraction policy: ${PSTRAIN_FP_CONTRACT_POLICY}")
  endif()
  add_compile_options("${PSTRAIN_FP_CONTRACT_FLAG}")
  add_compile_definitions(
    PSTRAIN_FP_CONTRACT_POLICY="${PSTRAIN_FP_CONTRACT_POLICY}")
  message(STATUS
    "Floating-point contraction policy: ${PSTRAIN_FP_CONTRACT_POLICY} (declared)")
  add_compile_options(/W3)
else()
  # Contraction must not inherit a compiler or architecture default.  FMA
  # availability differs across architectures, so "off" can support a
  # cross-architecture comparison; "fast" is reproducible only together with
  # a fixed compiler version and -march target.
  set(PSTRAIN_FP_CONTRACT_POLICY "off" CACHE INTERNAL
      "Declared floating-point contraction policy for native targets")
  check_c_compiler_flag("-ffp-contract=${PSTRAIN_FP_CONTRACT_POLICY}"
    PSTRAIN_COMPILER_SUPPORTS_FP_CONTRACT)
  if(NOT PSTRAIN_COMPILER_SUPPORTS_FP_CONTRACT)
    message(FATAL_ERROR
      "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION} does not support the declared "
      "floating-point contraction policy: ${PSTRAIN_FP_CONTRACT_POLICY}")
  endif()
  add_compile_options("-ffp-contract=${PSTRAIN_FP_CONTRACT_POLICY}")
  add_compile_definitions(
    PSTRAIN_FP_CONTRACT_POLICY="${PSTRAIN_FP_CONTRACT_POLICY}")
  message(STATUS
    "Floating-point contraction policy: ${PSTRAIN_FP_CONTRACT_POLICY} (declared)")
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

# PocketSphinx is a source dependency pinned to the official v5.1.1 commit.
# Prefer the nearby reference checkout used for development, but never trust
# its mutable worktree without checking both its commit and tracked state.
set(PSTRAIN_POCKETSPHINX_VERSION "5.1.1" CACHE INTERNAL
    "PocketSphinx version included in the pstrain measurement identity")
set(PSTRAIN_POCKETSPHINX_COMMIT
    "511126b492dcb267cf30d49d631946d7b61a9530" CACHE INTERNAL
    "PocketSphinx source commit")
get_filename_component(_pstrain_pocketsphinx_default
  "${CMAKE_CURRENT_SOURCE_DIR}/../../cmu/pocketsphinx" ABSOLUTE)
set(PSTRAIN_POCKETSPHINX_SOURCE_DIR "${_pstrain_pocketsphinx_default}" CACHE PATH
    "Optional local PocketSphinx v5.1.1 checkout; fetch the pinned commit if absent")

find_package(Git REQUIRED)
function(pstrain_verify_pocketsphinx_source source_dir)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${source_dir}"
    RESULT_VARIABLE _rev_result
    OUTPUT_VARIABLE _resolved_commit
    ERROR_VARIABLE _rev_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _rev_result EQUAL 0)
    message(FATAL_ERROR
      "PocketSphinx source at ${source_dir} is not a readable Git checkout: ${_rev_error}")
  endif()
  if(NOT _resolved_commit STREQUAL PSTRAIN_POCKETSPHINX_COMMIT)
    message(FATAL_ERROR
      "PocketSphinx source drift at ${source_dir}: expected v${PSTRAIN_POCKETSPHINX_VERSION} "
      "(${PSTRAIN_POCKETSPHINX_COMMIT}), found ${_resolved_commit}. Refusing to build.")
  endif()
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" status --porcelain=v1 --untracked-files=no
    WORKING_DIRECTORY "${source_dir}"
    RESULT_VARIABLE _status_result
    OUTPUT_VARIABLE _tracked_changes
    ERROR_VARIABLE _status_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _status_result EQUAL 0)
    message(FATAL_ERROR
      "Could not verify PocketSphinx tracked files at ${source_dir}: ${_status_error}")
  endif()
  if(NOT _tracked_changes STREQUAL "")
    message(FATAL_ERROR
      "PocketSphinx source at ${source_dir} has tracked modifications:\n${_tracked_changes}\n"
      "Refusing to build modified vendor source; restore official v${PSTRAIN_POCKETSPHINX_VERSION}.")
  endif()
  set(PSTRAIN_POCKETSPHINX_RESOLVED_COMMIT "${_resolved_commit}" PARENT_SCOPE)
endfunction()

if(IS_DIRECTORY "${PSTRAIN_POCKETSPHINX_SOURCE_DIR}")
  if(NOT EXISTS "${PSTRAIN_POCKETSPHINX_SOURCE_DIR}/.git")
    message(FATAL_ERROR
      "PocketSphinx source path exists but is not a Git checkout: "
      "${PSTRAIN_POCKETSPHINX_SOURCE_DIR}")
  endif()
  pstrain_verify_pocketsphinx_source("${PSTRAIN_POCKETSPHINX_SOURCE_DIR}")
  message(STATUS "Using verified local PocketSphinx source: ${PSTRAIN_POCKETSPHINX_SOURCE_DIR}")
  FetchContent_Declare(pocketsphinx_source
    SOURCE_DIR "${PSTRAIN_POCKETSPHINX_SOURCE_DIR}")
else()
  message(STATUS
    "Local PocketSphinx source absent at ${PSTRAIN_POCKETSPHINX_SOURCE_DIR}; fetching pinned v${PSTRAIN_POCKETSPHINX_VERSION}")
  FetchContent_Declare(pocketsphinx_source
    GIT_REPOSITORY "https://github.com/cmusphinx/pocketsphinx.git"
    GIT_TAG "${PSTRAIN_POCKETSPHINX_COMMIT}"
    GIT_SHALLOW FALSE
    GIT_PROGRESS TRUE)
endif()
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()
FetchContent_GetProperties(pocketsphinx_source)
if(NOT pocketsphinx_source_POPULATED)
  FetchContent_Populate(pocketsphinx_source)
endif()
pstrain_verify_pocketsphinx_source("${pocketsphinx_source_SOURCE_DIR}")
set(PSTRAIN_POCKETSPHINX_RESOLVED_COMMIT
    "${PSTRAIN_POCKETSPHINX_RESOLVED_COMMIT}" CACHE INTERNAL
    "Verified PocketSphinx source commit used by this build" FORCE)
message(STATUS
  "PocketSphinx provenance: v${PSTRAIN_POCKETSPHINX_VERSION} (${PSTRAIN_POCKETSPHINX_RESOLVED_COMMIT})")

# Configure only PocketSphinx's upstream C-library target.  Its top-level
# SKBUILD branch also builds Python bindings and installs acoustic models,
# neither of which belongs in pstrain's wheel.  Keeping its generated config
# in the dependency binary directory also avoids overwriting pstrain's own
# config.h (the upstream top level uses CMAKE_BINARY_DIR).
set(_pstrain_package_name "${PACKAGE_NAME}")
set(_pstrain_package_version "${PACKAGE_VERSION}")
set(_pstrain_package_string "${PACKAGE_STRING}")
set(PACKAGE_NAME "PocketSphinx")
set(PACKAGE_VERSION "${PSTRAIN_POCKETSPHINX_VERSION}")
set(PACKAGE_STRING "PocketSphinx ${PSTRAIN_POCKETSPHINX_VERSION}")
set(PACKAGE_TARNAME "pocketsphinx")
set(PACKAGE_URL "https://github.com/cmusphinx/pocketsphinx")
set(PACKAGE_BUGREPORT "dhdaines@gmail.com")
set(DEFAULT_RADIX 12)
set(FIXED_POINT OFF)
include(CheckCSourceCompiles)
check_c_source_compiles("_Thread_local int x; int main(void) { x = 42; return x; }"
  PSTRAIN_PS_HAVE_C11_THREAD_LOCAL)
check_c_source_compiles("__thread int x; int main(void) { x = 42; return x; }"
  PSTRAIN_PS_HAVE_GCC_THREAD_LOCAL)
if(PSTRAIN_PS_HAVE_C11_THREAD_LOCAL OR PSTRAIN_PS_HAVE_GCC_THREAD_LOCAL)
  set(PS_USE_THREAD_LOCAL_RNG ON)
endif()
file(MAKE_DIRECTORY "${pocketsphinx_source_BINARY_DIR}/include/pocketsphinx")
configure_file("${pocketsphinx_source_SOURCE_DIR}/config.h.in"
  "${pocketsphinx_source_BINARY_DIR}/config.h")
configure_file("${pocketsphinx_source_SOURCE_DIR}/sphinx_config.h.in"
  "${pocketsphinx_source_BINARY_DIR}/include/pocketsphinx/sphinx_config.h")

# Use the checked-in generated parser and scanner on every platform.  This is
# what upstream does when suitable generators are absent and avoids writes to
# the verified source checkout during a build.
set(_pstrain_disable_bison "${CMAKE_DISABLE_FIND_PACKAGE_BISON}")
set(_pstrain_disable_flex "${CMAKE_DISABLE_FIND_PACKAGE_FLEX}")
set(CMAKE_DISABLE_FIND_PACKAGE_BISON TRUE)
set(CMAKE_DISABLE_FIND_PACKAGE_FLEX TRUE)
add_subdirectory("${pocketsphinx_source_SOURCE_DIR}/src"
  "${pocketsphinx_source_BINARY_DIR}/src")
get_target_property(_pstrain_pocketsphinx_type pocketsphinx TYPE)
if(NOT _pstrain_pocketsphinx_type STREQUAL "SHARED_LIBRARY")
  message(FATAL_ERROR
    "PocketSphinx must be shared when building the bundled fallback; configured target type: "
    "${_pstrain_pocketsphinx_type}")
endif()
message(STATUS "PocketSphinx target type: ${_pstrain_pocketsphinx_type}")
set(CMAKE_DISABLE_FIND_PACKAGE_BISON "${_pstrain_disable_bison}")
set(CMAKE_DISABLE_FIND_PACKAGE_FLEX "${_pstrain_disable_flex}")
set(PACKAGE_NAME "${_pstrain_package_name}")
set(PACKAGE_VERSION "${_pstrain_package_version}")
set(PACKAGE_STRING "${_pstrain_package_string}")

target_include_directories(pocketsphinx BEFORE PUBLIC
  "${pocketsphinx_source_SOURCE_DIR}/include"
  "${pocketsphinx_source_BINARY_DIR}/include"
  PRIVATE "${pocketsphinx_source_BINARY_DIR}"
)
target_compile_definitions(pocketsphinx PRIVATE HAVE_CONFIG_H)
if(WIN32)
  # Mirror PocketSphinx's own top-level shared-library build. Its export.h
  # selects dllexport while src/CMakeLists.txt supplies POCKETSPHINX_EXPORTS.
  # Keep SPHINX_DLL private: pstrain intentionally compiles overlapping
  # sphinxbase sources, so applying upstream's dllimport side to those sources
  # produces __imp_* references to definitions in the same target.
  target_compile_definitions(pocketsphinx PRIVATE SPHINX_DLL)
endif()
set_target_properties(pocketsphinx PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  VERSION "${PSTRAIN_POCKETSPHINX_VERSION}"
  SOVERSION "${PSTRAIN_POCKETSPHINX_VERSION}"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
if(UNIX AND NOT APPLE)
  # Prevent ELF symbol interposition with the intentionally vendored older
  # sphinxbase implementation exported by libpstrainc.
  target_link_options(pocketsphinx PRIVATE "LINKER:-Bsymbolic")
endif()
install(TARGETS pocketsphinx
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" NAMELINK_SKIP
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

# Output directories (for development builds)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# The wheel installs libraries under pstrain/_lib/lib and programs under
# pstrain/_lib/bin.  Give every installed native consumer a wheel-relative
# search path so auditwheel and delocate can locate the bundled PocketSphinx
# library without access to the temporary build tree.
if(APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path/../lib")
elseif(UNIX)
  set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
endif()

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
