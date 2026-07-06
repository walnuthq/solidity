# Find MLIR and LLVM packages
#
# This module defines:
#  MLIR_FOUND - system has MLIR
#  MLIR_INCLUDE_DIRS - the MLIR include directories
#  MLIR_LIBRARIES - libraries needed to use MLIR
#  MLIR_DEFINITIONS - compiler flags needed to use MLIR

if(NOT DEFINED LLVM_PATH)
    # Try to find LLVM automatically
    find_package(LLVM QUIET CONFIG)
    if(LLVM_FOUND)
        set(LLVM_PATH ${LLVM_DIR})
    else()
        # Common installation paths
        set(LLVM_SEARCH_PATHS
            /opt/homebrew/opt/llvm@21
            /opt/homebrew/opt/llvm@20
            /opt/homebrew/opt/llvm@19
            /opt/homebrew/opt/llvm@18
            /opt/homebrew/opt/llvm
            /usr/local/opt/llvm
            /usr/lib/llvm-17
            /usr/lib/llvm-16
            /usr/lib/llvm-15
            /usr/lib/llvm-14
        )
        
        foreach(path ${LLVM_SEARCH_PATHS})
            if(EXISTS ${path})
                set(LLVM_PATH ${path})
                break()
            endif()
        endforeach()
    endif()
endif()

if(LLVM_PATH)
    set(CMAKE_PREFIX_PATH ${LLVM_PATH} ${CMAKE_PREFIX_PATH})
    find_package(LLVM REQUIRED CONFIG)
    
    if(LLVM_FOUND)
        message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
        message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
        
        # Find MLIR within LLVM
        find_package(MLIR QUIET CONFIG)
        
        if(NOT MLIR_FOUND)
            # MLIR might be part of LLVM installation
            set(MLIR_DIR ${LLVM_DIR}/../mlir)
            find_package(MLIR QUIET CONFIG)
            
            if(NOT MLIR_FOUND)
                # Try alternative paths
                set(MLIR_DIR ${LLVM_DIR}/../../mlir)
                find_package(MLIR QUIET CONFIG)
            endif()
        endif()
        
        if(MLIR_FOUND)
            message(STATUS "Found MLIR")
            
            # Set up MLIR variables
            set(MLIR_INCLUDE_DIRS ${LLVM_INCLUDE_DIRS})
            set(MLIR_DEFINITIONS ${LLVM_DEFINITIONS})
            
            # Core MLIR libraries including dialects needed for Solidity
            set(MLIR_LIBRARIES
                MLIRIR
                MLIRSupport
                MLIRParser
                MLIRPass
                MLIRTransforms
                MLIRAnalysis
                MLIRDialect
                MLIROptLib
                MLIRFuncDialect
                MLIRArithDialect
                MLIRControlFlowDialect
                MLIRMemRefDialect
                MLIRSCFDialect
            )
            
            # Add LLVM support libraries
            list(APPEND MLIR_LIBRARIES
                LLVMCore
                LLVMSupport
            )
        else()
            message(WARNING "MLIR not found in LLVM installation")
            set(MLIR_FOUND FALSE)
        endif()
    endif()
else()
    message(WARNING "LLVM/MLIR not found. MLIR support will be disabled.")
    set(MLIR_FOUND FALSE)
endif()

# Export the variables
if(MLIR_FOUND)
    set(MLIR_FOUND TRUE CACHE BOOL "MLIR found")
    set(MLIR_INCLUDE_DIRS ${MLIR_INCLUDE_DIRS} CACHE PATH "MLIR include directories")
    set(MLIR_LIBRARIES ${MLIR_LIBRARIES} CACHE STRING "MLIR libraries")
    set(MLIR_DEFINITIONS ${MLIR_DEFINITIONS} CACHE STRING "MLIR definitions")
else()
    set(MLIR_FOUND FALSE CACHE BOOL "MLIR not found")
endif()