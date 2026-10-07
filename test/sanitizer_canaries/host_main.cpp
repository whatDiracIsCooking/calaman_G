// host_main.cpp -- `sanitizer_canary_host <name>`: one deliberate host-side
// bug in calaman code, for ASan / LSan / UBSan to report. Exits 0 when the bug
// goes unnoticed, so only the sanitizer can make the run fail; 2 on a bad
// argument. See docs/sanitizers.md (S1, S3, G4).
#include <climits>
#include <cstdio>
#include <cstring>
#include <thread>

namespace calaman::canary {

namespace {

// ASan: writes one element past a heap allocation. The volatile index keeps
// the compiler from proving the store out of bounds and dropping it.
void heap_buffer_overflow() {
  volatile int past_end = 8;
  int *const buffer = new int[8];
  buffer[past_end] = 1;
  delete[] buffer;
}

// LSan: drops the only pointer to a heap allocation. The leak's stack is
// calaman's own, so a vendor-scoped lsan.supp line must not swallow it. It
// runs on a thread of its own: LSan scans the main thread's stack and
// registers conservatively at exit, and a stale copy of the pointer there
// (codegen-dependent; seen in #234) keeps the allocation reachable.
void leak() {
  std::thread([] {
    int *volatile leaked = new int[64];
    leaked[0] = 1;
    leaked = nullptr;
  }).join();
}

// UBSan: signed int overflow. The volatile operand keeps it a runtime add.
void signed_overflow() {
  volatile int top = INT_MAX;
  volatile int sum = top + 1;
  (void)sum;
}

} // namespace

} // namespace calaman::canary

int main(const int argc, char **const argv) {
  if (argc == 2 && std::strcmp(argv[1], "heap-buffer-overflow") == 0) {
    calaman::canary::heap_buffer_overflow();
    return 0;
  }
  if (argc == 2 && std::strcmp(argv[1], "leak") == 0) {
    calaman::canary::leak();
    return 0;
  }
  if (argc == 2 && std::strcmp(argv[1], "signed-overflow") == 0) {
    calaman::canary::signed_overflow();
    return 0;
  }
  std::fputs("usage: sanitizer_canary_host "
             "<heap-buffer-overflow|leak|signed-overflow>\n",
             stderr);
  return 2;
}
