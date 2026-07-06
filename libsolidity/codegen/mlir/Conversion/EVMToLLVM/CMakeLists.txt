# evm dialect -> LLVM dialect conversion (M5).

if(MLIR_FOUND)
    add_library(EVMToLLVMConversion STATIC
        EVMToLLVM.cpp
    )

    target_include_directories(EVMToLLVMConversion PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${MLIR_INCLUDE_DIRS}
    )

    target_link_libraries(EVMToLLVMConversion PUBLIC
        EVMMLIRDialect
        MLIRArithToLLVM
        MLIRControlFlowToLLVM
        MLIRFuncToLLVM
        MLIRLLVMDialect
    )

    set_source_files_properties(
        EVMToLLVM.cpp
        PROPERTIES COMPILE_FLAGS "-Wno-error -w"
    )
endif()
