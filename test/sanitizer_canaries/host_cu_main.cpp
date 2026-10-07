// host_cu_main.cpp -- `sanitizer_canary_host_cu <name>`: runs one host-side
// defect in host_cu_canary.cu. Exits 0 when it goes unnoticed, so only the
// sanitizer can make the run fail; 2 on a bad argument. A binary of its own so
// sanitizer_canary_host stays free of the GPU runtime. See docs/sanitizers.md
// (G1, G4).
#include "host_cu_canary.h"

#include <cstdio>
#include <cstring>

int main(const int argc, char **const argv) {
  if (argc == 2 && std::strcmp(argv[1], "heap-buffer-overflow") == 0) {
    calaman::canary::cu_heap_buffer_overflow();
    return 0;
  }
  if (argc == 2 && std::strcmp(argv[1], "signed-overflow") == 0) {
    calaman::canary::cu_signed_overflow();
    return 0;
  }
  std::fputs("usage: sanitizer_canary_host_cu "
             "<heap-buffer-overflow|signed-overflow>\n",
             stderr);
  return 2;
}
