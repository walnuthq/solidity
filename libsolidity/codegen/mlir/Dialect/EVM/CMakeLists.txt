# MLIR EVM dialect (rung 3 of the dialect ladder) CMake configuration.

if(MLIR_FOUND)
    set(LLVM_TARGET_DEFINITIONS EVMDialect.td)

    include(${LLVM_DIR}/TableGen.cmake)

    set(MLIR_TABLEGEN_INCLUDES
        "-I${MLIR_INCLUDE_DIRS}"
        "-I${LLVM_INCLUDE_DIRS}"
    )

    tablegen(MLIR EVMOps.h.inc -gen-op-decls ${MLIR_TABLEGEN_INCLUDES})
    tablegen(MLIR EVMOps.cpp.inc -gen-op-defs ${MLIR_TABLEGEN_INCLUDES})
    tablegen(MLIR EVMDialect.h.inc -gen-dialect-decls ${MLIR_TABLEGEN_INCLUDES})
    tablegen(MLIR EVMDialect.cpp.inc -gen-dialect-defs ${MLIR_TABLEGEN_INCLUDES})

    add_custom_target(EVMOpsIncGen
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/EVMOps.h.inc
        ${CMAKE_CURRENT_BINARY_DIR}/EVMOps.cpp.inc
        ${CMAKE_CURRENT_BINARY_DIR}/EVMDialect.h.inc
        ${CMAKE_CURRENT_BINARY_DIR}/EVMDialect.cpp.inc
    )

    add_library(EVMMLIRDialect STATIC
        EVMDialect.cpp
        EVMOps.cpp
    )

    add_dependencies(EVMMLIRDialect EVMOpsIncGen)

    target_include_directories(EVMMLIRDialect PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${MLIR_INCLUDE_DIRS}
    )

    target_link_libraries(EVMMLIRDialect PUBLIC
        MLIRIR
        MLIRSupport
        MLIRParser
    )
    target_compile_definitions(EVMMLIRDialect PUBLIC ${MLIR_DEFINITIONS})
    target_compile_definitions(EVMMLIRDialect PUBLIC SOLIDITY_HAS_MLIR=1)

    set_source_files_properties(
        EVMDialect.cpp
        EVMOps.cpp
        PROPERTIES COMPILE_FLAGS "-Wno-error -w"
    )
endif()
