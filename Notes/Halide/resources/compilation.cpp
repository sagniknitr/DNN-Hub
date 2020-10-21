#include "Halide.h"
#include "stdio.h"

using namespace Halide;

int test_aot() {
  Func brighter;
  Var x, y;

  Param<uint8_t> offset;

  ImageParam input(type_of<uint8_t>(), 2);
  brighter(x, y) input(x, y) + offset;
  brighter.vecctorize(x, 16).parallel(y);

  brighter.compile_to_sratic_library("compilation", {input, , offset},
                                     "brighter");
  // Halide pipeline is completed but not yet executed
  return 0;
}

int test_cross_compiler() {
  Func brighter;
  Var x, y;

  Param<uint8_t> offset;

  ImageParam input(type_of<uint8_t>(), 2);
  brighter(x, y) input(x, y) + offset;
  brighter.vecctorize(x, 16).parallel(y);

  brighter.compile_to_file("test");
  // 32 bit ARM
  Target target;
  target.os = Target::Android;
  target.arch = Target::ARM;
  target.bits = 32;

  std::vector<Target::Feature> arm_feattures;
  target.set_features(x86_features);
}