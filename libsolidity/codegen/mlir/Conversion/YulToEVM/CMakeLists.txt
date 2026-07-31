# yul dialect -> evm dialect conversion (M4).

if(MLIR_FOUND)
    add_library(YulToEVMConversion STATIC
        YulToEVM.cpp
    )

    target_include_directories(YulToEVMConversion PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${MLIR_INCLUDE_DIRS}
    )

    target_link_libraries(YulToEVMConversion PUBLIC
        YulMLIRDialect
        EVMMLIRDialect
        MLIRArithDialect
        MLIRControlFlowDialect
        MLIRFuncDialect
        MLIRPass
        MLIRTransforms
    )

    set_source_files_properties(
        YulToEVM.cpp
        PROPERTIES COMPILE_FLAGS "-Wno-error -w"
    )
endif()
