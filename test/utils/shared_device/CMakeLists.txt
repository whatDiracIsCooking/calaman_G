# test/utils/shared_device/ -- the one-DeviceHandle-per-process accessor and the
# GoogleTest global environment that tears it down. A GPU test utility (it links
# GoogleTest), so it lives under utils/, not shared/ (see test/shared/README.md
# for the shared/-vs-utils/ split). The implementation unit includes
# <gtest/gtest.h> and registers the environment; GTest::gtest is PRIVATE because
# the interface exposes none of it -- it exports only shared_device(), whose one
# public type is DeviceHandle (re-exported from
# calaman.test.shared.device_handle).
calaman_add_cxx_module_library(
  NAME
  calaman.test.utils.shared_device
  PRIMARY_INTERFACE
  shared_device.cppm
  IMPLEMENTATION
  shared_device.cpp
  LINK_PUBLIC
  calaman::test::shared::device_handle
  LINK_PRIVATE
  GTest::gtest
  IMPORT_STD
)
