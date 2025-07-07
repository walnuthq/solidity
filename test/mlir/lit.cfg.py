# -*- Python -*-

import os
import platform
import lit.formats
import lit.util

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'Solidity-MLIR'

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir', '.sol']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.solidity_obj_root, 'test', 'mlir')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

# Propagate some variables from the host environment.
llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

# Add solc to the path
config.substitutions.append(('%solc', config.solc_executable))
config.substitutions.append(('%FileCheck', config.filecheck_executable))

# Add MLIR tools
config.substitutions.append(('%mlir-opt', os.path.join(config.llvm_tools_dir, 'mlir-opt')))
config.substitutions.append(('%mlir-translate', os.path.join(config.llvm_tools_dir, 'mlir-translate')))

# Exclude some directories from testing
config.excludes = ['Inputs', 'CMakeFiles']

# Add features
if config.solidity_has_mlir:
    config.available_features.add('mlir')
EOF < /dev/null