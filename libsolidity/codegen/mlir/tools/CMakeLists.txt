# Development / CI tools for the MLIR pipeline.

if(MLIR_FOUND)
    add_executable(yul-dialect-test yul-dialect-test.cpp)

    target_link_libraries(yul-dialect-test PRIVATE
        YulTextTarget
        YulMLIRDialect
        yul
        langutil
        solutil
        MLIRParser
    )

    set_source_files_properties(
        yul-dialect-test.cpp
        PROPERTIES COMPILE_FLAGS "-Wno-error -w"
    )

    add_executable(yul-import-test yul-import-test.cpp)

    target_link_libraries(yul-import-test PRIVATE
        YulASTImport
        YulTextTarget
        YulMLIRDialect
        yul
        langutil
        solutil
        MLIRParser
    )

    set_source_files_properties(
        yul-import-test.cpp
        PROPERTIES COMPILE_FLAGS "-Wno-error -w"
    )

    add_executable(evm-lower-test evm-lower-test.cpp)

    target_link_libraries(evm-lower-test PRIVATE
        YulToEVMConversion
        YulASTImport
        YulMLIRDialect
        EVMMLIRDialect
        yul
        langutil
        solutil
        MLIRParser
    )

    set_source_files_properties(
        evm-lower-test.cpp
        PROPERTIES COMPILE_FLAGS "-Wno-error -w"
    )

    add_executable(sol-lower-test sol-lower-test.cpp)

    target_link_libraries(sol-lower-test PRIVATE
        SolToYulConversion
        YulToEVMConversion
        YulTextTarget
        SolidityDialect
        YulMLIRDialect
        EVMMLIRDialect
        yul
        langutil
        solutil
        MLIRParser
    )

    set_source_files_properties(
        sol-lower-test.cpp
        PROPERTIES COMPILE_FLAGS "-Wno-error -w"
    )

    add_executable(evm-riscv-test evm-riscv-test.cpp)

    # LLVM symbols come exclusively through the monolithic dylib (see
    # Target/RISCV/CMakeLists.txt) - do not add LLVM static archives here.
    target_link_libraries(evm-riscv-test PRIVATE
        EVMToLLVMConversion
        RISCVTarget
        YulToEVMConversion
        YulASTImport
        YulMLIRDialect
        EVMMLIRDialect
        yul
        langutil
        solutil
        MLIRExecutionEngine
        MLIRParser
    )

    find_program(EVM_LLD_PATH ld.lld
        PATHS ${LLVM_TOOLS_BINARY_DIR} /opt/homebrew/opt/lld/bin /usr/local/opt/lld/bin)
    if(NOT EVM_LLD_PATH)
        set(EVM_LLD_PATH "ld.lld")
    endif()

    target_compile_definitions(evm-riscv-test PRIVATE
        LLVM_BIN_DIR="${LLVM_TOOLS_BINARY_DIR}"
        LLD_PATH="${EVM_LLD_PATH}"
        EVM_RT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../runtime/evm-rt"
    )

    set_source_files_properties(
        evm-riscv-test.cpp
        PROPERTIES COMPILE_FLAGS "-Wno-error -w"
    )

    add_executable(erhi-host-test erhi-host-test.cpp)

    target_link_libraries(erhi-host-test PRIVATE
        EVMToLLVMConversion
        RISCVTarget
        YulToEVMConversion
        YulASTImport
        YulMLIRDialect
        EVMMLIRDialect
        yul
        langutil
        solutil
        MLIRExecutionEngine
        MLIRParser
    )

    set_source_files_properties(
        erhi-host-test.cpp
        PROPERTIES COMPILE_FLAGS "-Wno-error -w"
    )

    add_executable(yul2rv yul2rv.cpp)

    target_link_libraries(yul2rv PRIVATE
        EVMToLLVMConversion
        RISCVTarget
        YulToEVMConversion
        YulASTImport
        YulMLIRDialect
        EVMMLIRDialect
        yul
        langutil
        solutil
        MLIRParser
    )

    set_source_files_properties(
        yul2rv.cpp
        PROPERTIES COMPILE_FLAGS "-Wno-error -w"
    )
endif()
