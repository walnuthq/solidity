# EVM assembly target: lowers the evm dialect to libevmasm Assembly.

if(MLIR_FOUND)
    add_library(EVMAsmTarget STATIC
        EVMAssemblyEmitter.cpp
        EVMPipeline.cpp
    )

    target_include_directories(EVMAsmTarget PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${MLIR_INCLUDE_DIRS}
    )

    target_link_libraries(EVMAsmTarget PUBLIC
        YulASTImport
        YulToEVMConversion
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
        EVMPipeline.cpp
        PROPERTIES COMPILE_FLAGS "-Wno-error -w"
    )
endif()
