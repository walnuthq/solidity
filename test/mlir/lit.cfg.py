# -*- Python -*-

import os
import platform
import lit.formats
import lit.util

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'Solidity-MLIR'

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(execute_external=True)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir', '.sol']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.solidity_obj_root, 'test', 'mlir')

# Add tools dir to PATH so lit can find FileCheck, etc.
path = os.path.pathsep.join([config.llvm_tools_dir, config.environment.get('PATH', '')])
config.environment['PATH'] = path

# Propagate some variables from the host environment.
for var in ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP']:
    val = os.environ.get(var)
    if val:
        config.environment[var] = val

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
