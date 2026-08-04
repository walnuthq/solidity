# Lit configuration for the MLIR ladder.
#
# Each test pins the IR shape at a rung. Behaviour is covered separately by the
# differential harnesses in the parent directory - the two answer different
# questions, and a lowering that changes shape without changing behaviour is
# exactly what this catches and they do not.

import os
import lit.formats

config.name = "solc-mlir"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".sol", ".yul", ".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.environ.get(
    "MLIR_LIT_OUTPUT_DIR", os.path.join(config.test_source_root, "Output")
)

tools = os.environ.get("MLIR_TOOL_DIR", "")
filecheck = os.environ.get("FILECHECK", "FileCheck")

config.substitutions.append(("%sol2evm", os.path.join(tools, "sol2evm")))
config.substitutions.append(("%yul2evm", os.path.join(tools, "yul2evm")))
config.substitutions.append(("%FileCheck", filecheck))
