# Script half of calaman_add_kernel_name_check(): list every STO_ENTRY symbol
# (a `__global__` entry) in each archive's cubins and fail on any whose mangled
# name lacks `calaman`, or when no entry is found at all (cuobjdump reading
# nothing would otherwise pass vacuously).
#
# Run via:
#   cmake -DCUOBJDUMP=<cuobjdump> -DARCHIVES=<a.a|b.a> -DSTAMP=<file> \
#     -P calaman_check_kernel_names.cmake

if(NOT CUOBJDUMP
   OR NOT ARCHIVES
   OR NOT STAMP
)
  message(FATAL_ERROR "CUOBJDUMP, ARCHIVES and STAMP must be set")
endif()
string(REPLACE "|" ";" _archives "${ARCHIVES}")

set(_entries 0)
set(_offenders "")
foreach(_archive IN LISTS _archives)
  execute_process(
    COMMAND "${CUOBJDUMP}" -symbols "${_archive}"
    OUTPUT_VARIABLE _listing
    ERROR_VARIABLE _err
    RESULT_VARIABLE _rc
  )
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "cuobjdump -symbols ${_archive} failed:\n${_err}")
  endif()
  string(REPLACE "\n" ";" _lines "${_listing}")
  foreach(_line IN LISTS _lines)
    if(_line MATCHES "STO_ENTRY[ \t]+([^ \t]+)")
      set(_symbol "${CMAKE_MATCH_1}")
      math(EXPR _entries "${_entries} + 1")
      if(NOT _symbol MATCHES "calaman")
        list(APPEND _offenders "${_symbol} (${_archive})")
      endif()
    endif()
  endforeach()
endforeach()

if(_entries EQUAL 0)
  message(FATAL_ERROR "No __global__ entry found in any device archive.")
endif()
if(_offenders)
  list(JOIN _offenders "\n  " _report)
  message(
    FATAL_ERROR
      "These kernels do not mangle 'calaman', so compute-sanitizer's racecheck"
      " and synccheck filter (--kernel-name kns=calaman) would skip them. Put"
      " the kernel, or the functor it is instantiated with, in calaman::"
      " (docs/sanitizers.md, S2):\n  ${_report}"
  )
endif()
file(WRITE "${STAMP}" "${_entries} entries checked\n")
