# sol dialect -> yul dialect conversion (M2).

if(MLIR_FOUND)
    add_library(SolToYulConversion STATIC
        SolToYul.cpp
    )

    target_include_directories(SolToYulConversion PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${MLIR_INCLUDE_DIRS}
    )

    target_link_libraries(SolToYulConversion PUBLIC
        SolidityDialect
        YulMLIRDialect
        # Inline assembly is imported rather than re-parsed by hand.
        YulASTImport
        solutil
    )

    set_source_files_properties(
        SolToYul.cpp
        PROPERTIES COMPILE_FLAGS "-Wno-error -w"
    )
endif()
