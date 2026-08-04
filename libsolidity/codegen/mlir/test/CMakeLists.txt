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
            ENVIRONMENT "MLIR_TOOL_DIR=${mlirToolDir};MLIR_LIT_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/lit"
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

		# The same reach check starting at Solidity rather than legacy Yul. This
		# small all-green directory is a regression gate; the full semantic corpus
		# is measured with the same script as a deliberate wider sweep.
		add_test(
			NAME mlir-sol-coverage
			COMMAND ${Python3_EXECUTABLE} ${mlirTestDir}/sol_corpus_coverage.py
				--sol2evm ${mlirToolDir}/sol2evm
				--corpus ${CMAKE_SOURCE_DIR}/test/libsolidity/semanticTests/arithmetics
				--require-all-bytecode
		)
		set_tests_properties(mlir-sol-coverage PROPERTIES LABELS mlir)

        # Integration: unlike the rung-shape tools above, this enters through
        # the real compiler CLI and proves --mlir-bin reaches the production
        # Yul-IR -> MLIR backend independently of the legacy compatibility tool.
        add_test(
            NAME mlir-solc-integration
            COMMAND ${Python3_EXECUTABLE} ${mlirTestDir}/solc_integration.py
                --solc ${solcBinary}
                --sol2evm ${mlirToolDir}/sol2evm
                --source ${mlirTestDir}/lit/create_contract.sol
        )
        set_tests_properties(mlir-solc-integration PROPERTIES LABELS mlir)

        # Behaviour at the sol rung: compile the same Solidity through both
        # backends, deploy both creation objects, and compare every selector.
        add_test(
            NAME mlir-sol-differential
            COMMAND ${Python3_EXECUTABLE} ${mlirTestDir}/sol_differential.py
                --solc ${solcBinary}
                --sol2evm ${mlirToolDir}/sol2evm
                --source ${mlirTestDir}/lit/builtins_and_packed.sol
        )
        set_tests_properties(mlir-sol-differential PROPERTIES
            SKIP_RETURN_CODE 77
            LABELS mlir
        )

        add_test(
            NAME mlir-sol-assembly-differential
            COMMAND ${Python3_EXECUTABLE} ${mlirTestDir}/sol_differential.py
                --solc ${solcBinary}
                --sol2evm ${mlirToolDir}/sol2evm
                --source ${mlirTestDir}/lit/inline_assembly.sol
        )
		set_tests_properties(mlir-sol-assembly-differential PROPERTIES
			SKIP_RETURN_CODE 77
			LABELS mlir
		)

		add_test(
			NAME mlir-sol-external-call-differential
			COMMAND ${Python3_EXECUTABLE} ${mlirTestDir}/sol_differential.py
				--solc ${solcBinary}
				--sol2evm ${mlirToolDir}/sol2evm
				--source ${mlirTestDir}/lit/external_call.sol
		)
		set_tests_properties(mlir-sol-external-call-differential PROPERTIES
			SKIP_RETURN_CODE 77
			LABELS mlir
		)

		add_test(
			NAME mlir-sol-memory-differential
			COMMAND ${Python3_EXECUTABLE} ${mlirTestDir}/sol_differential.py
				--solc ${solcBinary}
				--sol2evm ${mlirToolDir}/sol2evm
				--source ${mlirTestDir}/lit/memory_aggregates.sol
		)
		set_tests_properties(mlir-sol-memory-differential PROPERTIES
			SKIP_RETURN_CODE 77
			LABELS mlir
		)

		add_test(
			NAME mlir-sol-storage-array-differential
			COMMAND ${Python3_EXECUTABLE} ${mlirTestDir}/sol_differential.py
				--solc ${solcBinary}
				--sol2evm ${mlirToolDir}/sol2evm
				--source ${mlirTestDir}/lit/storage_arrays.sol
		)
		set_tests_properties(mlir-sol-storage-array-differential PROPERTIES
			SKIP_RETURN_CODE 77
			LABELS mlir
		)

		add_test(
			NAME mlir-sol-user-value-type-differential
			COMMAND ${Python3_EXECUTABLE} ${mlirTestDir}/sol_differential.py
				--solc ${solcBinary}
				--sol2evm ${mlirToolDir}/sol2evm
				--source ${mlirTestDir}/lit/user_defined_value_types.sol
		)
		set_tests_properties(mlir-sol-user-value-type-differential PROPERTIES
			SKIP_RETURN_CODE 77
			LABELS mlir
		)

		add_test(
			NAME mlir-sol-short-circuit-differential
			COMMAND ${Python3_EXECUTABLE} ${mlirTestDir}/sol_differential.py
				--solc ${solcBinary}
				--sol2evm ${mlirToolDir}/sol2evm
				--source ${mlirTestDir}/lit/short_circuit.sol
		)
		set_tests_properties(mlir-sol-short-circuit-differential PROPERTIES
			SKIP_RETURN_CODE 77
			LABELS mlir
		)

		add_test(
			NAME mlir-sol-overload-differential
			COMMAND ${Python3_EXECUTABLE} ${mlirTestDir}/sol_differential.py
				--solc ${solcBinary}
				--sol2evm ${mlirToolDir}/sol2evm
				--source ${mlirTestDir}/lit/overloads.sol
		)
		set_tests_properties(mlir-sol-overload-differential PROPERTIES
			SKIP_RETURN_CODE 77
			LABELS mlir
		)

		add_test(
			NAME mlir-sol-concat-differential
			COMMAND ${Python3_EXECUTABLE} ${mlirTestDir}/sol_differential.py
				--solc ${solcBinary}
				--sol2evm ${mlirToolDir}/sol2evm
				--source ${mlirTestDir}/lit/concat.sol
		)
		set_tests_properties(mlir-sol-concat-differential PROPERTIES
			SKIP_RETURN_CODE 77
			LABELS mlir
		)

		add_test(
			NAME mlir-sol-precompile-differential
			COMMAND ${Python3_EXECUTABLE} ${mlirTestDir}/sol_differential.py
				--solc ${solcBinary}
				--sol2evm ${mlirToolDir}/sol2evm
				--source ${mlirTestDir}/lit/precompiles.sol
		)
		set_tests_properties(mlir-sol-precompile-differential PROPERTIES
			SKIP_RETURN_CODE 77
			LABELS mlir
		)

		add_test(
			NAME mlir-sol-library-call-differential
			COMMAND ${Python3_EXECUTABLE} ${mlirTestDir}/sol_differential.py
				--solc ${solcBinary}
				--sol2evm ${mlirToolDir}/sol2evm
				--source ${mlirTestDir}/lit/library_call.sol
		)
		set_tests_properties(mlir-sol-library-call-differential PROPERTIES
			SKIP_RETURN_CODE 77
			LABELS mlir
		)

		add_test(
			NAME mlir-sol-loop-differential
			COMMAND ${Python3_EXECUTABLE} ${mlirTestDir}/sol_differential.py
				--solc ${solcBinary}
				--sol2evm ${mlirToolDir}/sol2evm
				--source ${mlirTestDir}/lit/loops.sol
		)
		set_tests_properties(mlir-sol-loop-differential PROPERTIES
			SKIP_RETURN_CODE 77
			LABELS mlir
		)

		add_test(
			NAME mlir-sol-low-level-call-differential
			COMMAND ${Python3_EXECUTABLE} ${mlirTestDir}/sol_differential.py
				--solc ${solcBinary}
				--sol2evm ${mlirToolDir}/sol2evm
				--source ${mlirTestDir}/lit/low_level_call.sol
		)
		set_tests_properties(mlir-sol-low-level-call-differential PROPERTIES
			SKIP_RETURN_CODE 77
			LABELS mlir
		)

		add_test(
			NAME mlir-sol-abi-array-differential
			COMMAND ${Python3_EXECUTABLE} ${mlirTestDir}/sol_differential.py
				--solc ${solcBinary}
				--sol2evm ${mlirToolDir}/sol2evm
				--source ${mlirTestDir}/lit/abi_arrays.sol
		)
		set_tests_properties(mlir-sol-abi-array-differential PROPERTIES
			SKIP_RETURN_CODE 77
			LABELS mlir
		)

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
