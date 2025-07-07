# MLIR Solidity Dialect CMake configuration

# Find MLIR (optional for now)
find_package(MLIR QUIET)

if(MLIR_FOUND)
    message(STATUS "Building with MLIR support")
    
    # Set up TableGen processing for the Solidity dialect
    set(LLVM_TARGET_DEFINITIONS SolidityDialect.td)
    
    # Include TableGen support from LLVM
    include(${LLVM_DIR}/TableGen.cmake)
    
    # Generate operation declarations and definitions from TableGen
    # We need to provide include paths to MLIR's TableGen files
    set(MLIR_TABLEGEN_INCLUDES 
        "-I${MLIR_INCLUDE_DIRS}"
        "-I${LLVM_INCLUDE_DIRS}"
    )
    
    tablegen(MLIR SolidityOps.h.inc -gen-op-decls ${MLIR_TABLEGEN_INCLUDES})
    tablegen(MLIR SolidityOps.cpp.inc -gen-op-defs ${MLIR_TABLEGEN_INCLUDES})
    tablegen(MLIR SolidityDialect.h.inc -gen-dialect-decls ${MLIR_TABLEGEN_INCLUDES})
    tablegen(MLIR SolidityDialect.cpp.inc -gen-dialect-defs ${MLIR_TABLEGEN_INCLUDES})
    # Skip type generation for now - causing mlir-tblgen crash
    # tablegen(MLIR SolidityTypes.h.inc -gen-typedef-decls ${MLIR_TABLEGEN_INCLUDES})
    # tablegen(MLIR SolidityTypes.cpp.inc -gen-typedef-defs ${MLIR_TABLEGEN_INCLUDES})
    
    # Create custom target for TableGen outputs
    add_custom_target(SolidityOpsIncGen
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/SolidityOps.h.inc
        ${CMAKE_CURRENT_BINARY_DIR}/SolidityOps.cpp.inc
        ${CMAKE_CURRENT_BINARY_DIR}/SolidityDialect.h.inc
        ${CMAKE_CURRENT_BINARY_DIR}/SolidityDialect.cpp.inc
        # Removed type files from dependencies
    )
    
    # Create the Solidity dialect library with source files
    add_library(SolidityDialect STATIC
        SolidityDialect.cpp
        SolidityOps.cpp
    )
    
    # Add dependency on TableGen generated files
    add_dependencies(SolidityDialect SolidityOpsIncGen)
    
    # Include directory for generated files
    target_include_directories(SolidityDialect PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    
    # Link with MLIR libraries
    target_link_libraries(SolidityDialect PUBLIC 
        ${MLIR_LIBRARIES}
        MLIRIR
        MLIRSupport
        MLIRTableGen
    )
    target_include_directories(SolidityDialect PUBLIC ${MLIR_INCLUDE_DIRS})
    target_compile_definitions(SolidityDialect PUBLIC ${MLIR_DEFINITIONS})
    target_compile_definitions(SolidityDialect PUBLIC SOLIDITY_HAS_MLIR=1)
    
    # Disable warnings for MLIR dialect files
    set_source_files_properties(
        SolidityDialect.cpp
        SolidityOps.cpp
        PROPERTIES COMPILE_FLAGS "-Wno-error -w"
    )
    
    # Export the dialect availability
    set(SOLIDITY_HAS_MLIR TRUE PARENT_SCOPE)
    
else()
    message(STATUS "Building without MLIR support - using simplified implementation")
    
    # Create a stub library for compatibility
    add_library(SolidityDialect INTERFACE)
    
    set(SOLIDITY_HAS_MLIR FALSE PARENT_SCOPE)
endif()