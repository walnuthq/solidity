find_program(LIT_EXECUTABLE NAMES lit llvm-lit)

if(LIT_EXECUTABLE)
    add_custom_target(check-solidity-lit
        COMMAND
            ${LIT_EXECUTABLE}
            -sv
            ${CMAKE_CURRENT_SOURCE_DIR}
            --param solc=$<TARGET_FILE:solc>
        DEPENDS solc
        COMMENT "Running Solidity lit tests"
        USES_TERMINAL
    )
endif()
