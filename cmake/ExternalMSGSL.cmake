# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

FetchContent_Declare(
    msgsl
    GIT_REPOSITORY https://github.com/microsoft/GSL.git
    GIT_TAG        0f6dbc9e2915ef5c16830f3fa3565738de2a9230 # 3.1.0
)
FetchContent_GetProperties(msgsl)

if(NOT msgsl_POPULATED)
    FetchContent_Populate(msgsl)

    set(GSL_CXX_STANDARD "14" CACHE STRING "" FORCE)
    set(GSL_TEST OFF CACHE BOOL "" FORCE)
    mark_as_advanced(GSL_CXX_STANDARD )
    mark_as_advanced(GSL_TEST)
    mark_as_advanced(FETCHCONTENT_SOURCE_DIR_MSGSL)
    mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_MSGSL)

    add_subdirectory(
        ${msgsl_SOURCE_DIR}
        EXCLUDE_FROM_ALL)
endif()
