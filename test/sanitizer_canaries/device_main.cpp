// device_main.cpp -- `sanitizer_canary_device <name>`: runs one canary kernel
// from device_canaries.cu. Exits 0 whatever the kernel did, so only the
// compute-sanitizer launcher can make the run fail; 2 on a bad argument.
#include "device_canaries.h"

#include <cstdio>

int main(const int argc, char **const argv) {
  if (argc != 2 || !calaman::canary::run_device_canary(argv[1])) {
    std::fputs("usage: sanitizer_canary_device <memcheck|memcheck-leak|"
               "initcheck|racecheck-error|racecheck-warning|synccheck>\n",
               stderr);
    return 2;
  }
  return 0;
}
