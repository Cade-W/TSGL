/*
 * testRectangle.cpp
 *
 * Usage: ./testRectangle
 */

#include <tsgl.h>
#include <cmath>

using namespace tsgl;

void rectangleFunction(Canvas& can) {
    ColorGLfloat colors[] = { ColorGLfloat(0.5,0.5,0.5,0.8), ColorGLfloat(0,0,1,0.8),
        ColorGLfloat(0,1,0,0.8), ColorGLfloat(0,1,1,0.8), ColorGLfloat(1,0,0,0.8),
        ColorGLfloat(1,0,1,0.8), ColorGLfloat(1,1,0,0.8), ColorGLfloat(1,1,1,0.8),
        ColorGLfloat(0.5,0.5,0.5,0.8), ColorGLfloat(0.5,0.5,1,0.8),
        ColorGLfloat(0.5,1,0.5,0.8), ColorGLfloat(0.5,1,1,0.8), ColorGLfloat(1,0.5,0.5,0.8),
        ColorGLfloat(1,0.5,1,0.8), ColorGLfloat(1,1,0.5,0.8), ColorGLfloat(0,0,0.5,0.8),
        ColorGLfloat(0,0.5,0,0.8), ColorGLfloat(0,0.5,0.5,0.8), ColorGLfloat(0.5,0,0,0.8),
        ColorGLfloat(0.5,0,0.5,0.8), ColorGLfloat(0.5,0.5,0,0.8), ColorGLfloat(0.5,0.5,0.5,0.8)};
    Rectangle * rectangle = new Rectangle(0,0,0,1,2,0,0,0,colors/* ColorGLfloat(1,0,0,1) */);
    // rectangle->setCenterX(2);
    // rectangle->setRotationPoint(0,0,0);
    can.add(rectangle);
    float floatVal = 0.0f;
    GLfloat delta = 0.05;
    while (can.isOpen()) {
        can.sleep();
        // rectangle->setCenterX(sin(floatVal/90));
        // rectangle->setCenterY(sin(floatVal/90));
        // rectangle->setCenterZ(sin(floatVal/90));
        // rectangle->setYaw(floatVal);
        // rectangle->setPitch(floatVal);
        // rectangle->setRoll(floatVal);
        // rectangle->setWidth(sin(floatVal/90) + 1);
        // rectangle->setHeight(sin(floatVal/90) + 2);
        // if (rectangle->getWidth() > 2 || rectangle->getWidth() < 1) {
        //     delta *= -1;
        //     // rectangle->setEdgeColor(ColorGLfloat(float(rand())/float((RAND_MAX)), float(rand())/float((RAND_MAX)), float(rand())/float((RAND_MAX)), 1));
        // }
        // rectangle->changeWidthBy(delta);
        // if (rectangle->getHeight() > 3 || rectangle->getHeight() < 1) {
        //     delta *= -1;
        //     // rectangle->setEdgeColor(ColorGLfloat(float(rand())/float((RAND_MAX)), float(rand())/float((RAND_MAX)), float(rand())/float((RAND_MAX)), 1));
        // }
        // rectangle->changeHeightBy(delta);
        // if (delta > 0) {
        //     rectangle->setColor(colors);
        // } else {
        //     rectangle->setColor(ColorGLfloat(1,0,0,1));
        // }
        floatVal += 1;
    }

    delete rectangle;
}

int main(int argc, char* argv[]) {
    int w = (argc > 1) ? atoi(argv[1]) : 0.9*Canvas::getDisplayHeight();
    int h = (argc > 2) ? atoi(argv[2]) : w;
    if (w <= 0 || h <= 0)     //Checked the passed width and height if they are valid
      w = h = 960;            //If not, set the width and height to a default value
    Canvas c(-1, -1, 1024, 620, "Basic Rectangle");
    c.setBackgroundColor(BLACK);
    c.run(rectangleFunction);
}