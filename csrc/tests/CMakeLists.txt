# C smoke tests for libpstrainc, registered with CTest.
#
# These link the static pstrainc library and exercise the real C entry points
# (currently libio model-file round-trips). Kept deliberately small: the
# heavy numerical coverage lives in the Python/CFFI test suite; this tier
# exists so the C-tests CI job fails when libpstrainc itself breaks.

add_executable(test_io_roundtrip test_io_roundtrip.c)
target_link_libraries(test_io_roundtrip pstrainc)
target_include_directories(test_io_roundtrip
  PRIVATE ${CMAKE_BINARY_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

# Run each test in its own working directory under the build tree so the
# temporary S3 files never touch the source tree.
add_test(
  NAME io_roundtrip
  COMMAND test_io_roundtrip ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(test_gauden_scaling test_gauden_scaling.c)
target_link_libraries(test_gauden_scaling pstrainc)
target_include_directories(test_gauden_scaling
  PRIVATE ${CMAKE_BINARY_DIR}
)

add_test(
  NAME gauden_scaling
  COMMAND test_gauden_scaling
)
