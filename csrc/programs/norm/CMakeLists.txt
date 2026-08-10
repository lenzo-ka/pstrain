set(PROGRAM norm)
set(SRCS
main.c
parse_cmd_ln.c
)

add_executable(${PROGRAM} ${SRCS})
target_link_libraries(${PROGRAM} pstrainc)
target_include_directories(
  ${PROGRAM} PRIVATE ${CMAKE_BINARY_DIR}
  ${PROGRAM} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROGRAM} PRIVATE ${CMAKE_SOURCE_DIR}/csrc
  ${PROGRAM} PUBLIC ${CMAKE_SOURCE_DIR}/csrc/include
  ${PROGRAM} PRIVATE ${CMAKE_SOURCE_DIR}/csrc
  )
install(TARGETS ${PROGRAM} RUNTIME DESTINATION ${PSTRAIN_INSTALL_PROGRAMS})
