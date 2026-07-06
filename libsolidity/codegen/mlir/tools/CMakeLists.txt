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
endif()
