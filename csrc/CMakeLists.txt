# pstrain C library and programs

# All library source files
set(LIB_SRCS
  # libio - file I/O
  libs/libio/s3gau_full_io.c
  libs/libio/s3map_io.c
  libs/libio/s3tmat_io.c
  libs/libio/segdmp.c
  libs/libio/model_def_io.c
  libs/libio/s3gau_io.c
  libs/libio/topo_read.c
  libs/libio/s3lamb_io.c
  libs/libio/s3phseg_io.c
  libs/libio/s3cb2mllr_io.c
  libs/libio/swap.c
  libs/libio/s3acc_io.c
  libs/libio/s3mixw_io.c
  libs/libio/pset_io.c
  libs/libio/s3ts2cb_io.c
  libs/libio/corpus.c
  libs/libio/s3io.c

  # libsphinxbase/util - utilities
  libs/libsphinxbase/util/ckd_alloc.c
  libs/libsphinxbase/util/logmath.c
  libs/libsphinxbase/util/cmd_ln.c
  libs/libsphinxbase/util/bitvec.c
  libs/libsphinxbase/util/genrand.c
  libs/libsphinxbase/util/hash_table.c
  libs/libsphinxbase/util/strfuncs.c
  libs/libsphinxbase/util/matrix.c
  libs/libsphinxbase/util/pio.c
  libs/libsphinxbase/util/heap.c
  libs/libsphinxbase/util/case.c
  libs/libsphinxbase/util/err.c
  libs/libsphinxbase/util/glist.c
  libs/libsphinxbase/util/dtoa.c
  libs/libsphinxbase/util/bitarr.c
  libs/libsphinxbase/util/profile.c
  libs/libsphinxbase/util/bio.c
  libs/libsphinxbase/util/filename.c
  libs/libsphinxbase/util/priority_queue.c
  libs/libsphinxbase/util/errno.c
  libs/libsphinxbase/util/listelem_alloc.c
  libs/libsphinxbase/util/mmio.c

  # libsphinxbase/feat - feature processing
  libs/libsphinxbase/feat/cmn.c
  libs/libsphinxbase/feat/agc.c
  libs/libsphinxbase/feat/lda.c
  libs/libsphinxbase/feat/feat.c
  libs/libsphinxbase/feat/cmn_live.c

  # libsphinxbase/fe - front-end (MFCC etc)
  libs/libsphinxbase/fe/fe_warp_inverse_linear.c
  libs/libsphinxbase/fe/fe_sigproc.c
  libs/libsphinxbase/fe/yin.c
  libs/libsphinxbase/fe/fe_interface.c
  libs/libsphinxbase/fe/fe_warp_affine.c
  libs/libsphinxbase/fe/fixlog.c
  libs/libsphinxbase/fe/fe_warp.c
  libs/libsphinxbase/fe/fe_prespch_buf.c
  libs/libsphinxbase/fe/fe_noise.c
  libs/libsphinxbase/fe/fe_warp_piecewise_linear.c

  # libclust - clustering
  libs/libclust/kmeans.c
  libs/libclust/div.c
  libs/libclust/kdtree.c
  libs/libclust/metric.c

  # libmllr - MLLR adaptation
  libs/libmllr/mllr_io.c
  libs/libmllr/mllr.c

  # libmodinv - model inventory
  libs/libmodinv/mod_inv.c
  libs/libmodinv/gauden.c

  # libcommon - common utilities
  libs/libcommon/state_seq.c
  libs/libcommon/quest.c
  libs/libcommon/vector.c
  libs/libcommon/best_q.c
  libs/libcommon/ck_seg.c
  libs/libcommon/cvt2triphone.c
  libs/libcommon/mk_phone_list.c
  libs/libcommon/phone_graph.c
  libs/libcommon/phone_graph_triphone.c
  libs/libcommon/state_seq_graph.c
  libs/libcommon/remap.c
  libs/libcommon/heap.c
  libs/libcommon/itree.c
  libs/libcommon/acmod_set.c
  libs/libcommon/mk_trans_seq.c
  libs/libcommon/mk_ts2ci.c
  libs/libcommon/mk_phone_seq.c
  libs/libcommon/was_added.c
  libs/libcommon/dtree.c
  libs/libcommon/prunetree_order.c
  libs/libcommon/ts2cb.c
  libs/libcommon/mk_sseq.c
  libs/libcommon/lexicon.c

  # libpstrain - pstrain helper functions
  libs/libpstrain/pstrain_cffi.c
  libs/libpstrain/pstrain_rng.c
  libs/libpstrain/pstrain_fe.c
  libs/libpstrain/pstrain_session.c
  libs/libpstrain/pstrain_bw.c
  libs/libpstrain/pstrain_flat.c
  libs/libpstrain/pstrain_inc_comp.c
  libs/libpstrain/pstrain_kmeans.c
  libs/libpstrain/pstrain_mdef_gen.c
  libs/libpstrain/pstrain_dtree.c
  libs/libpstrain/pstrain_agg_seg.c
  libs/libpstrain/pstrain_param_cnt.c
  libs/libpstrain/pstrain_kdtree.c
  libs/libpstrain/pstrain_delint.c
  libs/libpstrain/pstrain_map_adapt.c
  libs/libpstrain/pstrain_align.c
  libs/libpstrain/pstrain_decode.c

  # sphinx3 forced aligner (vendored from sphinx3, formerly under
  # programs/sphinx3_align). Compiled into the library so pstrain_align
  # can drive it in-process; main_align.c's main() is excluded via
  # PSTRAIN_LIBRARY_BUILD.
  programs/sphinx3_align/adaptor.c
  programs/sphinx3_align/approx_cont_mgau.c
  programs/sphinx3_align/ascr.c
  programs/sphinx3_align/cb2mllr_io.c
  programs/sphinx3_align/cmu6_lts_rules.c
  programs/sphinx3_align/cont_mgau.c
  programs/sphinx3_align/corpus.c
  programs/sphinx3_align/dict.c
  programs/sphinx3_align/dict2pid.c
  programs/sphinx3_align/fast_algo_struct.c
  programs/sphinx3_align/fillpen.c
  programs/sphinx3_align/gs.c
  programs/sphinx3_align/hmm.c
  programs/sphinx3_align/interp.c
  programs/sphinx3_align/kbcore.c
  programs/sphinx3_align/kdtree.c
  programs/sphinx3_align/logs3.c
  programs/sphinx3_align/lts.c
  programs/sphinx3_align/main_align.c
  programs/sphinx3_align/mdef.c
  programs/sphinx3_align/mllr.c
  programs/sphinx3_align/ms_gauden.c
  programs/sphinx3_align/ms_mgau.c
  programs/sphinx3_align/ms_mllr.c
  programs/sphinx3_align/ms_senone.c
  programs/sphinx3_align/s2_semi_mgau.c
  programs/sphinx3_align/s3_align.c
  programs/sphinx3_align/subvq.c
  programs/sphinx3_align/tmat.c
  programs/sphinx3_align/vector.c

  # BW training (from programs/bw, included as library code)
  programs/bw/accum.c
  programs/bw/backward.c
  programs/bw/baum_welch.c
  programs/bw/forward.c
  programs/bw/next_utt_states.c
  programs/bw/viterbi.c

  # mk_mdef_gen helpers (for CD mdef generation)
  programs/mk_mdef_gen/mk_mdef_gen.c
  programs/mk_mdef_gen/heap.c
  programs/mk_mdef_gen/hash.c

  # make_quests helpers (for phonetic question generation)
  programs/make_quests/main.c

  # tiestate helpers (for state tying)
  programs/tiestate/main.c

  # prunetree helpers (for tree pruning)
  programs/prunetree/main.c

  # agg_seg helpers (for segment aggregation)
  programs/agg_seg/main.c
  programs/agg_seg/agg_all_seg.c
  programs/agg_seg/agg_phn_seg.c
  programs/agg_seg/agg_st_seg.c
  programs/agg_seg/cnt_phn_seg.c
  programs/agg_seg/cnt_st_seg.c
  programs/agg_seg/mk_seg.c

  # param_cnt helpers (for parameter counting)
  programs/param_cnt/main.c
  programs/param_cnt/parse_cmd_ln.c
  programs/param_cnt/param_cnt.c
  programs/param_cnt/enum_corpus.c
  programs/param_cnt/ts_cnt.c
  programs/param_cnt/cb_cnt.c
  programs/param_cnt/phone_cnt.c

  # delint helpers (for deleted interpolation)
  programs/delint/main.c
  programs/delint/interp_fn.c

  # map_adapt helpers (for MAP adaptation)
  programs/map_adapt/main.c
  programs/map_adapt/cmd_ln.c

)

# Internal LAPACK (used if system LAPACK not found)
set(LAPACK_SRCS
  libs/libsphinxbase/util/slamch.c
  libs/libsphinxbase/util/slapack_lite.c
  libs/libsphinxbase/util/blas_lite.c
  libs/libsphinxbase/util/f2c_lite.c
)

# Find dependencies
find_package(BLAS)
find_package(LAPACK)
find_library(MATH_LIBRARY m)

# Build the main static library
if(NOT LAPACK_FOUND)
  message(STATUS "System LAPACK not found, using internal version")
  add_library(pstrainc STATIC ${LIB_SRCS} ${LAPACK_SRCS})
else()
  add_library(pstrainc STATIC ${LIB_SRCS})
  target_link_libraries(pstrainc PUBLIC ${LAPACK_LIBRARIES})
endif()

# Also build shared library for development
if(NOT LAPACK_FOUND)
  add_library(pstrainc_shared SHARED ${LIB_SRCS} ${LAPACK_SRCS})
else()
  add_library(pstrainc_shared SHARED ${LIB_SRCS})
  target_link_libraries(pstrainc_shared PUBLIC ${LAPACK_LIBRARIES})
endif()
set_target_properties(pstrainc_shared PROPERTIES OUTPUT_NAME pstrainc)
if(WIN32)
  # Keep the DLL import archive at pstrainc.lib and give the static archive a
  # distinct name so single-output generators can represent both targets.
  set_target_properties(pstrainc PROPERTIES ARCHIVE_OUTPUT_NAME pstrainc_static)
endif()
if(WIN32)
  target_sources(pstrainc_shared PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/pstrain.exports.def")
  set_target_properties(pstrainc_shared PROPERTIES
    LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/pstrain.exports.def")
  if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    # The PocketSphinx import library exports compatibility routines that are
    # intentionally also provided by pstrain's vendored sphinxbase sources.
    # link.exe accepts these local definitions; give lld-link the equivalent
    # selection rule so clang-cl produces the same shared core.
    target_link_options(pstrainc_shared PRIVATE "LINKER:/FORCE:MULTIPLE")
  endif()
elseif(APPLE)
  set_target_properties(pstrainc_shared PROPERTIES
    INSTALL_RPATH "@loader_path"
    LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/pstrain.exports.macos")
  target_link_options(pstrainc_shared PRIVATE
    "LINKER:-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/pstrain.exports.macos")
elseif(UNIX)
  set_target_properties(pstrainc_shared PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/pstrain.exports.elf")
  target_link_options(pstrainc_shared PRIVATE
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/pstrain.exports.elf")
endif()

target_link_libraries(pstrainc PUBLIC pocketsphinx)
target_link_libraries(pstrainc_shared PUBLIC pocketsphinx)
target_compile_definitions(pstrainc PRIVATE
  PSTRAIN_POCKETSPHINX_VERSION="${PSTRAIN_POCKETSPHINX_VERSION}"
  PSTRAIN_POCKETSPHINX_COMMIT="${PSTRAIN_POCKETSPHINX_RESOLVED_COMMIT}")
target_compile_definitions(pstrainc_shared PRIVATE
  PSTRAIN_POCKETSPHINX_VERSION="${PSTRAIN_POCKETSPHINX_VERSION}"
  PSTRAIN_POCKETSPHINX_COMMIT="${PSTRAIN_POCKETSPHINX_RESOLVED_COMMIT}")

target_include_directories(pstrainc
  PRIVATE ${CMAKE_BINARY_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/programs/bw
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/programs/mk_mdef_gen
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/programs/make_quests
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/programs/tiestate
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/programs/prunetree
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/programs/agg_seg
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/programs/param_cnt
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/programs/delint
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/programs/map_adapt
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/programs/sphinx3_align
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(pstrainc_shared
  PRIVATE ${CMAKE_BINARY_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/programs/bw
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/programs/mk_mdef_gen
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/programs/make_quests
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/programs/tiestate
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/programs/prunetree
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/programs/agg_seg
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/programs/param_cnt
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/programs/delint
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/programs/map_adapt
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/programs/sphinx3_align
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Define PSTRAIN_LIBRARY_BUILD to exclude main() functions from program sources
target_compile_definitions(pstrainc PRIVATE PSTRAIN_LIBRARY_BUILD)
target_compile_definitions(pstrainc_shared PRIVATE PSTRAIN_LIBRARY_BUILD)

# Symbol-namespace the sphinx3_align translation units so they don't
# collide with the (similarly-named, differently-implemented) copies in
# libcommon/libclust/libmodinv. See s3a_namespace.h for the mapping.
set(S3A_SOURCES
  programs/sphinx3_align/vector.c
  programs/sphinx3_align/kdtree.c
  programs/sphinx3_align/ms_gauden.c
  programs/sphinx3_align/ms_mgau.c
  programs/sphinx3_align/ms_senone.c
  programs/sphinx3_align/cont_mgau.c
  programs/sphinx3_align/approx_cont_mgau.c
  programs/sphinx3_align/s2_semi_mgau.c
  programs/sphinx3_align/subvq.c
  programs/sphinx3_align/tmat.c
  programs/sphinx3_align/gs.c
  programs/sphinx3_align/kbcore.c
  programs/sphinx3_align/main_align.c
)
set(S3A_NAMESPACE_HEADER
  "${CMAKE_CURRENT_SOURCE_DIR}/programs/sphinx3_align/s3a_namespace.h")
if(MSVC)
  set(S3A_FORCE_INCLUDE "/FI${S3A_NAMESPACE_HEADER}")
else()
  set(S3A_FORCE_INCLUDE "-include;${S3A_NAMESPACE_HEADER}")
endif()
set_source_files_properties(${S3A_SOURCES} PROPERTIES
  COMPILE_OPTIONS "${S3A_FORCE_INCLUDE}")

if(MATH_LIBRARY)
  target_link_libraries(pstrainc PUBLIC ${MATH_LIBRARY})
  target_link_libraries(pstrainc_shared PUBLIC ${MATH_LIBRARY})
endif()

set_target_properties(pstrainc PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 0
  POSITION_INDEPENDENT_CODE ON
  C_VISIBILITY_PRESET default
)

# Install library and headers
include(GNUInstallDirs)
install(TARGETS pstrainc pstrainc_shared
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Build CLI programs for development/testing
if(BUILD_CLI)
  add_subdirectory(programs/agg_seg)
  add_subdirectory(programs/bldtree)
  add_subdirectory(programs/bw)
  add_subdirectory(programs/cdcn_norm)
  add_subdirectory(programs/cdcn_train)
  add_subdirectory(programs/cp_parm)
  add_subdirectory(programs/delint)
  add_subdirectory(programs/inc_comp)
  add_subdirectory(programs/init_gau)
  add_subdirectory(programs/init_mixw)
  add_subdirectory(programs/kdtree)
  add_subdirectory(programs/kmeans_init)
  add_subdirectory(programs/make_quests)
  add_subdirectory(programs/map_adapt)
  add_subdirectory(programs/mixw_interp)
  add_subdirectory(programs/mk_flat)
  add_subdirectory(programs/mk_mdef_gen)
  add_subdirectory(programs/mk_mllr_class)
  add_subdirectory(programs/mk_s2sendump)
  add_subdirectory(programs/mk_ts2cb)
  add_subdirectory(programs/mllr_solve)
  add_subdirectory(programs/mllr_transform)
  add_subdirectory(programs/norm)
  add_subdirectory(programs/param_cnt)
  add_subdirectory(programs/printp)
  add_subdirectory(programs/prunetree)
  add_subdirectory(programs/sphinx_cepview)
  add_subdirectory(programs/sphinx_fe)
  add_subdirectory(programs/sphinx3_align)
  add_subdirectory(programs/tiestate)
endif()

# C smoke tests (CTest). Guarded by BUILD_TESTING (CMake convention).
# Default ON for normal/dev/CI builds; OFF inside scikit-build-core wheel
# builds (SKBUILD), where the tests add nothing shippable. enable_testing()
# is idempotent. The supported entry point is the repository root.
if(SKBUILD)
  option(BUILD_TESTING "Build the C smoke tests" OFF)
else()
  option(BUILD_TESTING "Build the C smoke tests" ON)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
