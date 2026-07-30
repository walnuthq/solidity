# EVM assembly target: lowers the evm dialect to libevmasm Assembly.

if(MLIR_FOUND)
    add_library(EVMAsmTarget STATIC
        EVMAssemblyEmitter.cpp
    )

    target_include_directories(EVMAsmTarget PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${MLIR_INCLUDE_DIRS}
    )

    target_link_libraries(EVMAsmTarget PUBLIC
        EVMMLIRDialect
        MLIRArithDialect
        MLIRControlFlowDialect
        MLIRFuncDialect
        evmasm
        langutil
        solutil
    )

    set_source_files_properties(
        EVMAssemblyEmitter.cpp
        PROPERTIES COMPILE_FLAGS "-Wno-error -w"
    )
endif()
