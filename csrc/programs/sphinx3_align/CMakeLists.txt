set(PROGRAM sphinx3_align)

# All sphinx3_align sources are compiled into libpstrainc (see ../../CMakeLists.txt
# LIB_SRCS). The standalone CLI just provides a main() — every other symbol
# comes from pstrainc. main_align.c is built once with the library (main()
# excluded via PSTRAIN_LIBRARY_BUILD) and once here as the program entry point.
add_executable(${PROGRAM} main_align.c)
target_compile_definitions(${PROGRAM} PRIVATE)
target_link_libraries(${PROGRAM} pstrainc)
target_include_directories(${PROGRAM}
  PRIVATE ${CMAKE_BINARY_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_SOURCE_DIR}/csrc
  PUBLIC ${CMAKE_SOURCE_DIR}/csrc/include
)
install(TARGETS ${PROGRAM} RUNTIME DESTINATION ${PSTRAIN_INSTALL_PROGRAMS})
