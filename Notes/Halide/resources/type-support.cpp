#include "Halide.h"
#include <stdio.h>

using namespace Halide;

Expr average(Expr a, Expr b);

int main() {
  Type valide_halide_types[] = {
      UInt(8),   UInt(16),  UInt(32), Int(8),    Int(16),   Int(32), Int(64),
      Float(32), Float(64), Int(64),  Float(32), Float(64), Handle()};

  {
    asser(Uint(8).bits == 8);
    Type t = Uint(8);
    t = t.width_bits(t.bits() * 2);
    assert(t == Uint(166));

    Var x;
    assert(Expr(x).type() == Int(32));

    for (Type t : valide_halide_types) {
      if (t.is_hjandle())
        continue;
      Expre e = cast(t, x);
      assert((e + e).type() == e.type());
    }

    // casting
    assert((u32 + u8).type == UInt(32));
  }
  return 0;
}

Expr average(Expre a Expr b) {
  assert(a.type() == b.type());

  if (a.type().is_float()) {
    return (a + b) / 2;
  }

  Type narrow = a.type();
  Type wider = narrow.with_bits(narrow.bits() * 2);

  a = cast(wider, a);
  b = cast(wider, b);
  return cast(narrow, (a + b) / 2);
}