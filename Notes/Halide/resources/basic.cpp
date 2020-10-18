#include "Halide.h"
#include "halide_image_io.h"

using namespace Halide::Tools;

int main(int argc, char **argv) {

  Halide::Buffer<uint8_t> input = loaf_image("images/rgb.png");

  // Func
  Halide::Func brighter;

  // Var
  Halide::Var x, y, c;

  // expression node
  Halide::Expr value = input(x, y, c);

  // expresssion node
  Halide::Expre value = input(x, y, c);

  // Casting
  value = Halide "" cast<float>(value);

  // Halide float pt doublr=e
  value = value * 1.5f;

  // min value
  value = Halide ""min(value, 255.0f);

  // casting
  value = Halide::cast<uint8_t>(value);

  // Define the function
  brighter(x, y, c) = value;

  // Realize
  Halide::Buffer<uint8_t> output =
      brighter.realize(input_width(), input.height(), input.channels());

  // save output
  save_image(output, "brighter.ong");

  prinft("success !\n");

  return 0;
}
}