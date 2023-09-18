# Copyright (c) Microsoft Corporation. All rights reserved.  Licensed under the MIT license.

FetchContent_Declare(
    zlib
    GIT_REPOSITORY https://github.com/madler/zlib.git
    GIT_TAG        cacf7f1d4e3d44d871b605da3b647f07d718623f # 1.2.11
)
FetchContent_GetProperties(zlib)
if(NOT zlib_POPULATED)
    FetchContent_Populate(zlib)

    set(SKIP_INSTALL_ALL ON CACHE BOOL "" FORCE)
    mark_as_advanced(AMD64)
    mark_as_advanced(ASM686)
    mark_as_advanced(EXECUTABLE_OUTPUT_PATH)
    mark_as_advanced(CMAKE_INSTALL_PREFIX)
    mark_as_advanced(INSTALL_BIN_DIR)
    mark_as_advanced(INSTALL_INC_DIR)
    mark_as_advanced(INSTALL_LIB_DIR)
    mark_as_advanced(INSTALL_MAN_DIR)
    mark_as_advanced(INSTALL_PKGCONFIG_DIR)
    mark_as_advanced(LIBRARY_OUTPUT_PATH)
    mark_as_advanced(CMAKE_BACKWARDS_COMPATIBILITY)
    mark_as_advanced(ZLIB_BUILD_STATIC)
    mark_as_advanced(SKIP_INSTALL_ALL)
    mark_as_advanced(FETCHCONTENT_SOURCE_DIR_ZLIB)
    mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_ZLIB)

    # ZLIB has no VERSION given to project(), needs to suppress CMP0048 warning
    set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS TRUE CACHE INTERNAL "Suppress CMP0048 warning" FORCE)
    add_subdirectory(
        ${zlib_SOURCE_DIR}
        EXCLUDE_FROM_ALL)
endif()
