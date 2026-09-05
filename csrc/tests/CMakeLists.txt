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

add_executable(test_variance_floor_consumers test_variance_floor_consumers.c)
target_link_libraries(test_variance_floor_consumers pstrainc)
target_include_directories(test_variance_floor_consumers
  PRIVATE ${CMAKE_BINARY_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..
)
add_test(
  NAME variance_floor_consumers
  COMMAND test_variance_floor_consumers ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(test_state_seq_ownership test_state_seq_ownership.c)
target_link_libraries(test_state_seq_ownership pstrainc)
target_include_directories(test_state_seq_ownership
  PRIVATE ${CMAKE_BINARY_DIR}
)

add_test(
  NAME state_seq_ownership
  COMMAND test_state_seq_ownership
)

add_executable(test_decode_start_utt
  test_decode_start_utt.c
  ../libs/libpstrain/pstrain_decode.c
)
target_include_directories(test_decode_start_utt
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/stubs
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_compile_definitions(test_decode_start_utt PRIVATE
  PSTRAIN_POCKETSPHINX_VERSION="test"
  PSTRAIN_POCKETSPHINX_COMMIT="stub"
)

add_test(
  NAME decode_start_utt
  COMMAND test_decode_start_utt
)

add_executable(test_prunetree_order test_prunetree_order.c)
target_link_libraries(test_prunetree_order pstrainc)
target_include_directories(test_prunetree_order
  PRIVATE ${CMAKE_BINARY_DIR}
)

add_test(
  NAME prunetree_order
  COMMAND test_prunetree_order
)

add_executable(test_rng test_rng.c)
target_link_libraries(test_rng pstrainc)
add_test(NAME rng_golden_and_libc_cross_check COMMAND test_rng)

add_executable(test_kmeans_init_continuation test_kmeans_init_continuation.c)
target_link_libraries(test_kmeans_init_continuation pstrainc)

set(KMEANS_CONTINUATION_DIR ${CMAKE_CURRENT_BINARY_DIR}/kmeans_continuation)
add_test(
  NAME kmeans_init_continuation_setup
  COMMAND test_kmeans_init_continuation setup ${KMEANS_CONTINUATION_DIR}
)
set_tests_properties(kmeans_init_continuation_setup PROPERTIES
  FIXTURES_SETUP kmeans_continuation_input
)

add_test(
  NAME kmeans_init_continuation_run
  COMMAND kmeans_init
    -gthobj state
    -omoddeffn ${KMEANS_CONTINUATION_DIR}/mdef
    -ts2cbfn .cont.
    -tsoff 0
    -tscnt 2
    -segdmpdirs ${KMEANS_CONTINUATION_DIR}
    -segdmpfn obs.dump
    -segidxfn obs.idx
    -feat 1s_c
    -ceplen 1
    -stride 1
    -ntrial 5
    -ndensity 2
    -reest no
    -vartiethr 0
    -meanfn ${KMEANS_CONTINUATION_DIR}/means
)
set_tests_properties(kmeans_init_continuation_run PROPERTIES
  FIXTURES_REQUIRED kmeans_continuation_input
  FIXTURES_SETUP kmeans_continuation_output
)

add_test(
  NAME kmeans_init_rng_continuation
  COMMAND test_kmeans_init_continuation check ${KMEANS_CONTINUATION_DIR}
)
set_tests_properties(kmeans_init_rng_continuation PROPERTIES
  FIXTURES_REQUIRED kmeans_continuation_output
)

add_executable(test_bw_state_seq_build test_bw_state_seq_build.c)
target_link_libraries(test_bw_state_seq_build pstrainc)
target_include_directories(test_bw_state_seq_build
  PRIVATE ${CMAKE_BINARY_DIR}
)

set(BW_FIXTURE ${PROJECT_SOURCE_DIR}/tests/fixtures/multipron_final_state)
add_test(
  NAME bw_state_seq_build
  COMMAND test_bw_state_seq_build
    ${BW_FIXTURE}/model/mdef
    ${BW_FIXTURE}/model/means
    ${BW_FIXTURE}/model/variances
    ${BW_FIXTURE}/model/mixture_weights
    ${BW_FIXTURE}/model/transition_matrices
    ${BW_FIXTURE}/dictionary.dict
    ${BW_FIXTURE}/filler.dict
)

add_executable(test_init_mixw_cleanup test_init_mixw_cleanup.c)
target_link_libraries(test_init_mixw_cleanup pstrainc)

set(INIT_MIXW_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/init_mixw_cleanup)
file(MAKE_DIRECTORY ${INIT_MIXW_OUTPUT_DIR})
add_test(
  NAME init_mixw_cleanup
  COMMAND test_init_mixw_cleanup
    ${BW_FIXTURE}/model/mdef
    ${BW_FIXTURE}/model/mixture_weights
    ${BW_FIXTURE}/model/means
    ${BW_FIXTURE}/model/variances
    ${BW_FIXTURE}/model/transition_matrices
    ${INIT_MIXW_OUTPUT_DIR}
)

add_executable(test_byteorder test_byteorder.c)
target_link_libraries(test_byteorder pstrainc)
target_include_directories(test_byteorder
  PRIVATE ${CMAKE_BINARY_DIR}
)

add_test(
  NAME byteorder
  COMMAND test_byteorder ${CMAKE_CURRENT_BINARY_DIR}
)
