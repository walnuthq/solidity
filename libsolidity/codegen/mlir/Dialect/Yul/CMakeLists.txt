# MLIR Yul dialect (rung 2 of the dialect ladder) CMake configuration.
# Mirrors the Sol dialect's TableGen setup.

if(MLIR_FOUND)
    # Set up TableGen processing for the Yul dialect
    set(LLVM_TARGET_DEFINITIONS YulDialect.td)

    # Include TableGen support from LLVM
    include(${LLVM_DIR}/TableGen.cmake)

    set(MLIR_TABLEGEN_INCLUDES
        "-I${MLIR_INCLUDE_DIRS}"
        "-I${LLVM_INCLUDE_DIRS}"
    )

    tablegen(MLIR YulOps.h.inc -gen-op-decls ${MLIR_TABLEGEN_INCLUDES})
    tablegen(MLIR YulOps.cpp.inc -gen-op-defs ${MLIR_TABLEGEN_INCLUDES})
    tablegen(MLIR YulDialect.h.inc -gen-dialect-decls ${MLIR_TABLEGEN_INCLUDES})
    tablegen(MLIR YulDialect.cpp.inc -gen-dialect-defs ${MLIR_TABLEGEN_INCLUDES})

    add_custom_target(YulOpsIncGen
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/YulOps.h.inc
        ${CMAKE_CURRENT_BINARY_DIR}/YulOps.cpp.inc
        ${CMAKE_CURRENT_BINARY_DIR}/YulDialect.h.inc
        ${CMAKE_CURRENT_BINARY_DIR}/YulDialect.cpp.inc
    )

    add_library(YulMLIRDialect STATIC
        YulDialect.cpp
        YulOps.cpp
    )

    add_dependencies(YulMLIRDialect YulOpsIncGen)

    target_include_directories(YulMLIRDialect PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${MLIR_INCLUDE_DIRS}
    )

    target_link_libraries(YulMLIRDialect PUBLIC
        MLIRIR
        MLIRSupport
        MLIRParser
    )
    target_compile_definitions(YulMLIRDialect PUBLIC ${MLIR_DEFINITIONS})
    target_compile_definitions(YulMLIRDialect PUBLIC SOLIDITY_HAS_MLIR=1)

    # MLIR headers are not clean under solc's pedantic warning flags.
    set_source_files_properties(
        YulDialect.cpp
        YulOps.cpp
        PROPERTIES COMPILE_FLAGS "-Wno-error -w"
    )
endif()
