# experimental/ -- the unstable tier: spikes and sketches not yet promoted to
# src/. Added from the top-level CMakeLists.txt only under
# CALAMAN_BUILD_EXPERIMENTAL (OFF by default), so a default build never sees it.
# One directory per module, same shape as src/.
add_subdirectory(xor_delta)
