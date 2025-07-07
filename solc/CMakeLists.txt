set(libsolcli_sources
	CommandLineInterface.cpp CommandLineInterface.h
	CommandLineParser.cpp CommandLineParser.h
	Exceptions.h
)

add_library(solcli ${libsolcli_sources})
target_link_libraries(solcli PUBLIC solidity Boost::boost Boost::program_options)

set(sources main.cpp)

add_executable(solc ${sources})
target_link_libraries(solc PRIVATE solcli)

# Link MLIR libraries directly to solc executable if available
if(MLIR_FOUND)
    find_library(MLIR_IR_LIB MLIRIR HINTS /opt/homebrew/opt/llvm@20/lib)
    find_library(MLIR_FUNC_LIB MLIRFunc HINTS /opt/homebrew/opt/llvm@20/lib)
    find_library(MLIR_ARITH_LIB MLIRArith HINTS /opt/homebrew/opt/llvm@20/lib)
    find_library(MLIR_SUPPORT_LIB MLIRSupport HINTS /opt/homebrew/opt/llvm@20/lib)
    
    if(MLIR_IR_LIB AND MLIR_FUNC_LIB AND MLIR_ARITH_LIB AND MLIR_SUPPORT_LIB)
        target_link_libraries(solc PRIVATE ${MLIR_IR_LIB} ${MLIR_FUNC_LIB} ${MLIR_ARITH_LIB} ${MLIR_SUPPORT_LIB})
    endif()
endif()

include(GNUInstallDirs)
install(TARGETS solc DESTINATION "${CMAKE_INSTALL_BINDIR}")

if(SOLC_LINK_STATIC AND UNIX AND NOT APPLE)
	# Produce solc as statically linked binary (includes C/C++ standard libraries)
	# This is not supported on macOS, see
	# https://developer.apple.com/library/content/qa/qa1118/_index.html.
	set_target_properties(
		solc PROPERTIES
		LINK_FLAGS -static
		LINK_SEARCH_START_STATIC ON
		LINK_SEARCH_END_STATIC ON
	)
elseif(SOLC_STATIC_STDLIBS AND UNIX AND NOT APPLE)
	set_target_properties(
		solc PROPERTIES
		LINK_FLAGS "-static-libgcc -static-libstdc++"
	)
endif()
