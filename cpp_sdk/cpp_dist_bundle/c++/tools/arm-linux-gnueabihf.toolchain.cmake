# For cross compiling on 32-bit arm Linux using gcc-arm-linux-gnueabihf package:
# - install arm-linux-gnueabi-gcc package:
#   $ sudo apt-get install gcc-arm-linux-gnueabihf binutils-arm-linux-gnueabihf g++-arm-linux-gnueabihf
# - cross-compiling config
#   $ cmake -DCMAKE_TOOLCHAIN_FILE=../dynamorio/make/toolchain-arm32.cmake ../dynamorio
# You may have to set CMAKE_FIND_ROOT_PATH to point to the target enviroment, e.g.
# by passing -DCMAKE_FIND_ROOT_PATH=/usr/arm-linux-gnueabihf on Debian-like systems.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)
# If using a different target, set -DTARGET_ABI=<abi> on the command line.
# Some of our pre-built libraries (such as libelftc) assume gnueabihf.
# To support both arm-linux-gnueabi and arm-linux-gnueabi, we rely on
# CMAKE_C_LIBRARY_ARCHITECTURE for libelftc libraries selection.
# If CMAKE_C_LIBRARY_ARCHITECTURE is not set, users need manually set it
# to gnueabi for using gnueabi build of libelftc libraries.
if (NOT DEFINED TARGET_ABI)
  set(TARGET_ABI "linux-gnueabihf")
endif ()
# specify the cross compiler
SET(CMAKE_C_COMPILER   arm-${TARGET_ABI}-gcc)
SET(CMAKE_CXX_COMPILER arm-${TARGET_ABI}-g++)

# To build the tests, we need to set where the target environment containing
# the required library is. On Debian-like systems, this is
# /usr/arm-linux-gnueabihf/.
SET(CMAKE_FIND_ROOT_PATH "/usr/arm-${TARGET_ABI}")
# search for programs in the build host directories
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# for libraries and headers in the target directories
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Set additional variables.
# If we don't set some of these, CMake will end up using the host version.
# We want the full path, however, so we can pass EXISTS and other checks in
# the our CMake code.
find_program(GCC_FULL_PATH arm-${TARGET_ABI}-gcc)
if (NOT GCC_FULL_PATH)
  message(FATAL_ERROR "Cross-compiler arm-${TARGET_ABI}-gcc not found")
endif ()
get_filename_component(GCC_DIR ${GCC_FULL_PATH} PATH)
SET(CMAKE_LINKER        ${GCC_DIR}/arm-${TARGET_ABI}-ld       CACHE FILEPATH "linker")
SET(CMAKE_ASM_COMPILER  ${GCC_DIR}/arm-${TARGET_ABI}-as       CACHE FILEPATH "assembler")
SET(CMAKE_OBJCOPY       ${GCC_DIR}/arm-${TARGET_ABI}-objcopy  CACHE FILEPATH "objcopy")
SET(CMAKE_STRIP         ${GCC_DIR}/arm-${TARGET_ABI}-strip    CACHE FILEPATH "strip")
SET(CMAKE_CPP           ${GCC_DIR}/arm-${TARGET_ABI}-cpp      CACHE FILEPATH "cpp")