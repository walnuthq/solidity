# -*- Python -*-

import os
import shutil
import lit.formats

config.name = "Solidity"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = [".sol"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.test_source_root, ".lit")

solidity_src_root = os.path.abspath(os.path.join(config.test_source_root, "..", ".."))
default_solc = os.path.join(solidity_src_root, "build", "solc", "solc")

solc = lit_config.params.get("solc") or os.environ.get("SOLC") or default_solc
if not os.path.isabs(solc):
    solc = os.path.abspath(solc)
filecheck = lit_config.params.get("FileCheck") or shutil.which("FileCheck")

if not os.path.exists(solc):
    lit_config.fatal("solc not found; pass --param solc=/path/to/solc or set SOLC")

if not filecheck:
    lit_config.fatal("FileCheck not found; pass --param FileCheck=/path/to/FileCheck")

config.substitutions.append(("%solc", solc))
config.substitutions.append(("%FileCheck", filecheck))

path_entries = [os.path.dirname(filecheck), config.environment.get("PATH", "")]
config.environment["PATH"] = os.pathsep.join(entry for entry in path_entries if entry)

config.excludes = ["Inputs", ".lit", "CMakeFiles"]
