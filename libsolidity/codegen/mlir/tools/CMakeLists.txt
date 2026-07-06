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
endif()
