# ---------------------------------------------------------------
# Programmer(s): Fabio Durastante @ IAC-CNR
# ---------------------------------------------------------------
# SUNDIALS Copyright Start
# Copyright (c) 2002-2020, Lawrence Livermore National Security
# and Southern Methodist University.
# All rights reserved.
#
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
# SUNDIALS Copyright End
# ---------------------------------------------------------------
# PSCTOOLKIT tests for SUNDIALS CMake-based configuration.
# ---------------------------------------------------------------

### This is only set if running GUI - simply return first time enabled
if(PSCTOOLKIT_DISABLED)
  set(PSCTOOLKIT_DISABLED FALSE CACHE INTERNAL "GUI - now enabled" FORCE)
  return()
endif()

message(STATUS "Checking for PSCTOOLKIT support... ")

# --- Find PSCTOOLKIT and test it --- #
find_package(PSCTOOLKIT REQUIRED)

# If we have the PSCTOOLKIT libraries,
if(PSCTOOLKIT_FOUND)
  message(STATUS "Checking for PSCTOOLKIT support... OK")
  # sundials_config.h symbols
  set(SUNDIALS_PSCTOOLKIT TRUE)
else()
  message(STATUS "Checking for PSCTOOLKIT support... FAILED")
  # sundials_config.h symbols
  set(SUNDIALS_PSCTOOLKIT FALSE)
endif()
