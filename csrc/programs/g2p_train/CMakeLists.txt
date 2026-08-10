set(PROGRAM g2p_train)
set(SRCS
FstPathFinder.cpp
g2p_train.cpp
M2MFstAligner.cpp
main.cpp
  )

add_executable(${PROGRAM} ${SRCS})
target_link_libraries(${PROGRAM} pstrainc
  ${FST} ${FSTSCRIPT} ${FSTFAR} ${FSTFARSCRIPT} ${NGRAM})
target_include_directories(
  ${PROGRAM} PRIVATE ${CMAKE_BINARY_DIR}
  ${PROGRAM} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROGRAM} PUBLIC ${CMAKE_SOURCE_DIR}/csrc/include
  ${PROGRAM} PRIVATE ${CMAKE_SOURCE_DIR}/csrc
  ${PROGRAM} INTERFACE ${CMAKE_SOURCE_DIR}/csrc/include
  ${PROGRAM} PRIVATE ${CMAKE_SOURCE_DIR}/csrc
  )
install(TARGETS ${PROGRAM} RUNTIME DESTINATION ${PSTRAIN_INSTALL_PROGRAMS})
