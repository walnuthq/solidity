# RISC-V target emission (M5): LLVM dialect -> LLVM IR -> RV32IM objects.

if(MLIR_FOUND)
    add_library(RISCVTarget STATIC
        RISCVTargetEmitter.cpp
    )

    target_include_directories(RISCVTarget PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${MLIR_INCLUDE_DIRS}
    )

    # Homebrew's MLIR static libs reference the monolithic libLLVM dylib;
    # linking individual LLVM static archives alongside would put two LLVM
    # copies in the process (registry state splits, pass manager traps).
    # The dylib carries all targets including RISCV.
    target_link_libraries(RISCVTarget PUBLIC
        MLIRLLVMDialect
        MLIRTargetLLVMIRExport
        MLIRLLVMToLLVMIRTranslation
        MLIRBuiltinToLLVMIRTranslation
        LLVM
    )

    set_source_files_properties(
        RISCVTargetEmitter.cpp
        PROPERTIES COMPILE_FLAGS "-Wno-error -w"
    )
endif()
