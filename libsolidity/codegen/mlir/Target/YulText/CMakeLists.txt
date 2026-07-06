# Yul-text target: prints `yul` dialect modules as libyul-compatible source.

if(MLIR_FOUND)
    add_library(YulTextTarget STATIC
        YulTextEmitter.cpp
    )

    target_include_directories(YulTextTarget PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${MLIR_INCLUDE_DIRS}
    )

    target_link_libraries(YulTextTarget PUBLIC YulMLIRDialect)

    set_source_files_properties(
        YulTextEmitter.cpp
        PROPERTIES COMPILE_FLAGS "-Wno-error -w"
    )
endif()
