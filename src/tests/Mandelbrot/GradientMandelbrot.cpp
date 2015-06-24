/*
 * GradientMandelbrot.cpp
 *
 *  Created on: May 28, 2015
 *      Author: Chris Dilley
 */

#include "GradientMandelbrot.h"

GradientMandelbrot::GradientMandelbrot(unsigned threads, unsigned depth) : Mandelbrot(threads, depth) {
  myThreads = threads;
  myDepth = depth;
  myRedraw = true;
}

void GradientMandelbrot::draw(Cart& can) {
  while (myRedraw) {
    setRedraw(false);
    #pragma omp parallel num_threads(myThreads)
    {
      unsigned int nthreads = omp_get_num_threads();
      double blockstart = can.getCartHeight() / nthreads;
      unsigned int iterations;
      double smooth;
      for (unsigned int k = 0; k <= (can.getWindowHeight() / nthreads) && can.getIsOpen(); k++) {  // As long as we aren't trying to render off of the screen...
        long double row = blockstart * omp_get_thread_num() + can.getMinY() + can.getPixelHeight() * k;
        for (long double col = can.getMinX(); col <= can.getMaxX(); col += can.getPixelWidth()) {
          complex c(col, row);
          complex z(col, row);
          smooth = exp(-std::abs(z));
          iterations = 0;
          while (std::abs(z) < 2.0l && iterations != myDepth) {
            iterations++;
            z = z * z + c;
            smooth += exp(-std::abs(z));
          }
          smooth /= (myDepth + 1);
          float value = (float)iterations/myDepth;
          can.drawPoint(col, row, ColorHSV((float)smooth * 6.0f, 1.0f, value, 1.0f));
          if (myRedraw) break;
        }
        can.handleIO();
        if (myRedraw) break;
      }
    }
    while (can.getIsOpen() && !myRedraw)
      can.sleep();  //Removed the timer and replaced it with an internal timer in the Canvas class
  }
}

void GradientMandelbrot::setRedraw(bool newValue) {
    myRedraw = newValue;
}



