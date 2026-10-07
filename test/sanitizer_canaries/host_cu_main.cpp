// host_cu_main.cpp -- `sanitizer_canary_host_cu`: runs the host-side heap
// overflow in host_cu_canary.cu. Exits 0 when it goes unnoticed, so only ASan
// can make the run fail. A binary of its own so sanitizer_canary_host stays
// free of the GPU runtime. See docs/sanitizers.md (G1).
#include "host_cu_canary.h"

int main() {
  calaman::canary::cu_heap_buffer_overflow();
  return 0;
}
