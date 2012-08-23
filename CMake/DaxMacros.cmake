##=============================================================================
##
##  Copyright (c) Kitware, Inc.
##  All rights reserved.
##  See LICENSE.txt for details.
##
##  This software is distributed WITHOUT ANY WARRANTY; without even
##  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
##  PURPOSE.  See the above copyright notice for more information.
##
##  Copyright 2012 Sandia Corporation.
##  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
##  the U.S. Government retains certain rights in this software.
##
##=============================================================================

include(CMakeParseArguments)

# Utility to build a kit name from the current directory.
function(dax_get_kit_name kitvar)
  # Will this always work?  It should if ${CMAKE_CURRENT_SOURCE_DIR} is
  # built from ${Dax_SOURCE_DIR}.
  string(REPLACE "${Dax_SOURCE_DIR}/" "" dir_prefix ${CMAKE_CURRENT_SOURCE_DIR})
  string(REPLACE "/" "_" kit "${dir_prefix}")
  set(${kitvar} "${kit}" PARENT_SCOPE)
  # Optional second argument to get dir_prefix.
  if (${ARGC} GREATER 1)
    set(${ARGV1} "${dir_prefix}" PARENT_SCOPE)
  endif (${ARGC} GREATER 1)
endfunction(dax_get_kit_name)

# Builds a source file and an executable that does nothing other than
# compile the given header files.
function(dax_add_header_build_test name dir_prefix use_cuda)
  set(hfiles ${ARGN})
  if (use_cuda)
    set(suffix ".cu")
  else (use_cuda)
    set(suffix ".cxx")
  endif (use_cuda)
  set(cxxfiles)
  foreach (header ${ARGN})
    string(REPLACE "${CMAKE_CURRENT_BINARY_DIR}" "" header "${header}")
    get_filename_component(headername ${header} NAME_WE)
    set(src ${CMAKE_CURRENT_BINARY_DIR}/testing/TestBuild_${name}_${headername}${suffix})
    configure_file(${Dax_SOURCE_DIR}/CMake/TestBuild.cxx.in ${src} @ONLY)
    set(cxxfiles ${cxxfiles} ${src})
  endforeach (header)

  if (use_cuda)
    cuda_add_library(TestBuild_${name} ${cxxfiles} ${hfiles})
  else (use_cuda)
    add_library(TestBuild_${name} ${cxxfiles} ${hfiles})
    if(DAX_EXTRA_COMPILER_WARNINGS)
      set_target_properties(TestBuild_${name}
        PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS_WARN_EXTRA})
    endif(DAX_EXTRA_COMPILER_WARNINGS)
  endif (use_cuda)
  set_source_files_properties(${hfiles}
    PROPERTIES HEADER_FILE_ONLY TRUE
    )
endfunction()

# Declare a list of header files.  Will make sure the header files get
# compiled and show up in an IDE.
function(dax_declare_headers)
  set(options CUDA)
  set(oneValueArgs)
  set(multiValueArgs)
  cmake_parse_arguments(DAX_DH "${options}"
    "${oneValueArgs}" "${multiValueArgs}"
    ${ARGN}
    )
  set(hfiles ${DAX_DH_UNPARSED_ARGUMENTS})
  dax_get_kit_name(name dir_prefix)
  dax_add_header_build_test(
    "${name}" "${dir_prefix}" "${DAX_DH_CUDA}" ${hfiles}
    )
endfunction(dax_declare_headers)

# Declare unit tests, which should be in the same directory as a kit
# (package, module, whatever you call it).  Usage:
#
# dax_unit_tests(
#   SOURCES <source_list>
#   LIBRARIES <dependent_library_list>
#   )
function(dax_unit_tests)
  set(options CUDA)
  set(oneValueArgs)
  set(multiValueArgs SOURCES LIBRARIES)
  cmake_parse_arguments(DAX_UT
    "${options}" "${oneValueArgs}" "${multiValueArgs}"
    ${ARGN}
    )
  if (DAX_ENABLE_TESTING)
    dax_get_kit_name(kit)
    #we use UnitTests_kit_ so that it is an unique key to exclude from coverage
    set(test_prog UnitTests_kit_${kit})
    create_test_sourcelist(TestSources ${test_prog}.cxx ${DAX_UT_SOURCES})
    if (DAX_UT_CUDA)
      cuda_add_executable(${test_prog} ${TestSources})
    else (DAX_UT_CUDA)
      add_executable(${test_prog} ${TestSources})
      if(DAX_EXTRA_COMPILER_WARNINGS)
        set_target_properties(${test_prog}
          PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS_WARN_EXTRA})
      endif(DAX_EXTRA_COMPILER_WARNINGS)
    endif (DAX_UT_CUDA)
    target_link_libraries(${test_prog} ${DAX_UT_LIBRARIES})
    foreach (test ${DAX_UT_SOURCES})
      get_filename_component(tname ${test} NAME_WE)
      add_test(NAME ${tname}
        COMMAND ${test_prog} ${tname}
        )
    endforeach (test)
  endif (DAX_ENABLE_TESTING)
endfunction(dax_unit_tests)
