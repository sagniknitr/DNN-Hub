#include "Halide.h"
#include <stdio.h>
#include <algorithm>


using namespace Halide;



int main() {
    Var x("x"), y("y");

    {
        Func gradient("gradient");
        gradient(x,y) = x+ y;
        gradient.trace_stores();

        Buffer<int> output = gradient.realize(4,4);


    }

    {
        Func gradient("gradient_tiled");
        gradient(x,y) = x + y;
        gradient.trace_stores();

        Var x_outer, x_inner, y_outer, y_inner;
        gradient.split(x, x_outer, x_inner, 4);
        gradient.split(y, y_outer, y_inner, 4);

        gradient.reorder(x_inner, y_inner, x_outer, y_outer);

        Buffer<int> output = gradient.realize(8,8);
    }


    {
                Func gradient("gradient_tiled");
        gradient(x,y) = x + y;
        gradient.trace_stores();

        Var x_outer, x_iiner;
        gradient.split(x, x_outer, x_inner, 4);

        gradient.vectorize(x_inner);
        Buffer <int> output = gradient.relaize(8,4);

    }
    {

                        Func gradient("gradient_tiled");
        gradient(x,y) = x + y;
        gradient.trace_stores();

        Var x_outer, x_iiner;
        gradient.split(x, x_outer, x_inner, 4);

        gradient.unroll(x_inner);
        Buffer <int> output = gradient.relaize(8,4);

    }
}