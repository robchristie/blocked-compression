#include "test.hpp"
#include "compress.hpp"
#include "fmt/format.h"
#include <fmt/core.h>

int main() {
  auto c = Compress();
  // c.test_default();
  c.test_custom(1024, 1024, 3);
  return 0;
}
