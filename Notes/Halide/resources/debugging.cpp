#include "Halide.h"
#include "halide_image_io.h"

using namespace Halide;

int main(int argc, char **argv) {

  {
    Func gradient("gradient");
    Var x("X"), y("Y");
    gradients(x, y) = x + y;

    gradient.trace_store();

    Buffer<int> output - gradient.realize(8, 8);

    // Debugginh out to html file
    gradient.cmpile_to_lowered_stmt("gradient.html", {}, HTML);
  }
  {
    Func floatf(x, y) =
        sin(x) + print(cos(y), " <- this is coe(", y, ") when x = ", x);

    f.realize(4, 4);
  }

  return 0;
}
