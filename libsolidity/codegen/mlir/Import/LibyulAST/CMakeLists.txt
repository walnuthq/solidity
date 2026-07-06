# libyul-AST -> yul dialect importer (M2b).

if(MLIR_FOUND)
    add_library(YulASTImport STATIC
        YulASTImporter.cpp
    )

    target_include_directories(YulASTImport PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${MLIR_INCLUDE_DIRS}
    )

    target_link_libraries(YulASTImport PUBLIC YulMLIRDialect yul langutil solutil)

    set_source_files_properties(
        YulASTImporter.cpp
        PROPERTIES COMPILE_FLAGS "-Wno-error -w"
    )
endif()
