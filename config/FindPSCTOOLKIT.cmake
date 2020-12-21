# ------------------------------------------------------------------------------
# Programmer(s): Fabio Durastante @ IAC-CNR
# ------------------------------------------------------------------------------
# SUNDIALS Copyright Start
# Copyright (c) 2002-2020, Lawrence Livermore National Security
# and Southern Methodist University.
# All rights reserved.
#
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
# SUNDIALS Copyright End
# ------------------------------------------------------------------------------

# First we look for the pieces of PSCTOOLKIT that have been installed
# The core is represented by the PSBLAS library, if that is not found we stop
# and fail.

find_path(temp_PSCTOOLKIT_INCLUDE_DIR Make.inc.psblas ${PSCTOOLKIT_DIR}/include)
if (temp_PSCTOOLKIT_INCLUDE_DIR)
    set(PSCTOOLKIT_INCLUDE_DIR ${temp_PSCTOOLKIT_INCLUDE_DIR})
    MESSAGE(STATUS "Found PSCTOOLKIT (${PSCTOOLKIT_INCLUDE_DIR})")
endif()
unset(temp_PSCTOOLKIT_INCLUDE_DIR CACHE)

# Check for AMG4PSBLAS
if( EXISTS ${PSCTOOLKIT_INCLUDE_DIR}/Make.inc.amg4psblas )
    MESSAGE(STATUS "Found AMG4PSBLAS")
    SET(AMG4PSBLAS_FOUND TRUE)
else()
    SET(AMG4PSBLAS_FOUND FALSE)
endif()

# Check for AMG4PSBLAS-EXTENSION
if( EXISTS ${PSCTOOLKIT_INCLUDE_DIR}/Make.inc.amg-ext )
    MESSAGE(STATUS "Found AMG4PSBLAS-EXT")
    SET(AMG4PSBLAS-EXT_FOUND TRUE)
else()
    SET(AMG4PSBLAS-EXT_FOUND FALSE)
endif()

# Now we parse the Make.inc file to set the compilation variables
# We start with the PSBLAS Make.inc.psblas file, this has to be found so we
# check for nothing
set(regex "BLAS=.*")
file(STRINGS ${PSCTOOLKIT_INCLUDE_DIR}/Make.inc.psblas LINK_BLAS REGEX "${regex}")
set(regex "BLAS=")
string(REGEX REPLACE "${regex}" "" LINK_BLAS "${LINK_BLAS}")

set(regex "METIS_LIB=.*")
file(STRINGS ${PSCTOOLKIT_INCLUDE_DIR}/Make.inc.psblas LINK_METIS_LIB REGEX "${regex}")
set(regex "METIS_LIB=")
string(REGEX REPLACE "${regex}" "" LINK_METIS_LIB "${LINK_METIS_LIB}")

set(regex "AMD_LIB=.*")
file(STRINGS ${PSCTOOLKIT_INCLUDE_DIR}/Make.inc.psblas LINK_AMD_LIB REGEX "${regex}")
set(regex "AMD_LIB=")
string(REGEX REPLACE "${regex}" "" LINK_AMD_LIB "${LINK_AMD_LIB}")

set(regex "PSBFDEFINES=.*")
file(STRINGS ${PSCTOOLKIT_INCLUDE_DIR}/Make.inc.psblas PSBFDEFINES REGEX "${regex}")
set(regex "PSBFDEFINES=")
string(REGEX REPLACE "${regex}" "" PSBFDEFINES "${PSBFDEFINES}")
set(regex "-D")
string(REGEX REPLACE "${regex}" "" PSBFDEFINES "${PSBFDEFINES}")
separate_arguments(PSBFDEFINES)

set(regex "PSBCDEFINES=.*")
file(STRINGS ${PSCTOOLKIT_INCLUDE_DIR}/Make.inc.psblas PSBCDEFINES REGEX "${regex}")
set(regex "PSBCDEFINES=")
string(REGEX REPLACE "${regex}" "" PSBCDEFINES "${PSBCDEFINES}")
set(regex "-D")
string(REGEX REPLACE "${regex}" "" PSBCDEFINES "${PSBFDEFINES}")
separate_arguments(PSBCDEFINES)

set(LINK_PSBLAS -lgfortran -L${PSCTOOLKIT_DIR}/lib -lpsb_base -lpsb_util -lpsb_cbind -lpsb_krylov -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi)

set(LINKED_LIBRARIES "${LINK_BLAS} ${LINK_METIS_LIB} ${LINK_AMD_LIB} ${LINK_PSBLAS}")
set(PSBLAS_INCLUDE ${PSCTOOLKIT_INCLUDE_DIR}/)
set(PSBLAS_MODULES ${PSCTOOLKIT_DIR}/modules/)

MESSAGE(STATUS "Linked Libraries ${LINKED_LIBRARIES}")
MESSAGE(STATUS "Include Directory ${PSBLAS_INCLUDE}")
MESSAGE(STATUS "Module Directory ${PSBLAS_MODULES}")

SET(PSCTOOLKIT_FOUND TRUE)
