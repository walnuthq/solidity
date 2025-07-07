# MLIR Test Suite Configuration

if(SOLIDITY_HAS_MLIR)
    # Configure lit site configuration
    configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
    )

    # Add lit test suite
    add_lit_testsuite(check-solidity-mlir
        "Running Solidity MLIR tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS 
            solc
            FileCheck
            SolidityDialect
    )

    set_target_properties(check-solidity-mlir PROPERTIES FOLDER "Tests")

    # Add test files
    file(GLOB MLIR_TEST_FILES "*.mlir" "*.sol")
    
    # Copy test files to build directory
    foreach(test_file ${MLIR_TEST_FILES})
        configure_file(${test_file} ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
    endforeach()
endif()
EOF < /dev/null