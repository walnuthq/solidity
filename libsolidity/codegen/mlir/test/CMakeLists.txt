# Tests for the MLIR ladder, registered with CTest so a regression is visible
# without anyone remembering to run a script.
#
# Two kinds, answering different questions. The lit suite pins the shape of each
# lowering and needs nothing but the tools. The differential harnesses execute
# what was compiled and compare it against solc, so they need a JSON-RPC node;
# without one they report themselves skipped rather than failed, because a
# machine with no anvil is not a broken build.

if(MLIR_FOUND)
    find_package(Python3 COMPONENTS Interpreter)

    if(Python3_Interpreter_FOUND)
        set(mlirTestDir ${CMAKE_CURRENT_SOURCE_DIR})
        set(mlirToolDir ${CMAKE_BINARY_DIR}/libsolidity/codegen/mlir/tools)
        set(solcBinary ${CMAKE_BINARY_DIR}/solc/solc)

        add_test(
            NAME mlir-lit
            COMMAND ${mlirTestDir}/lit/run.sh
        )
        set_tests_properties(mlir-lit PROPERTIES
            ENVIRONMENT "MLIR_TOOL_DIR=${mlirToolDir}"
            LABELS mlir
        )

        # Reach: how far every Yul object of a real contract gets down the
        # ladder. Scoped to one in-repo directory so it stays quick; the full
        # semantic suite is a deliberate, separate run.
        add_test(
            NAME mlir-coverage
            COMMAND ${Python3_EXECUTABLE} ${mlirTestDir}/corpus_coverage.py
                --solc ${solcBinary}
                --yul2evm ${mlirToolDir}/yul2evm
                --corpus ${CMAKE_SOURCE_DIR}/test/libsolidity/semanticTests/arithmetics
        )
        set_tests_properties(mlir-coverage PROPERTIES LABELS mlir)

        add_test(
            NAME mlir-differential
            COMMAND ${Python3_EXECUTABLE} ${mlirTestDir}/evm_differential.py
                --corpus ${mlirTestDir}/corpus
                --solc ${solcBinary}
                --yul2evm ${mlirToolDir}/yul2evm
        )
        set_tests_properties(mlir-differential PROPERTIES
            SKIP_RETURN_CODE 77
            LABELS mlir
        )

        add_test(
            NAME mlir-deploy-differential
            COMMAND ${Python3_EXECUTABLE} ${mlirTestDir}/deploy_differential.py
                --solc ${solcBinary}
                --yul2evm ${mlirToolDir}/yul2evm
                --corpus ${CMAKE_SOURCE_DIR}/test/libsolidity/semanticTests/arithmetics
        )
        set_tests_properties(mlir-deploy-differential PROPERTIES
            SKIP_RETURN_CODE 77
            LABELS mlir
        )
    endif()
endif()
