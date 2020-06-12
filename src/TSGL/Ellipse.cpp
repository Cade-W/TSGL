#include "Ellipse.h"

namespace tsgl {

/*!
 * \brief Explicitly constructs a new monocolored filled or outlined Ellipse.
 * \details This function draws a Ellipse with the given center, radii, color, and outline color.
 *   \param x The x coordinate of the Ellipse's center.
 *   \param y The y coordinate of the Ellipse's center.
 *   \param xRadius The horizontal radius of the Ellipse in pixels.
 *   \param yRadius The vertical radius of the Ellipse in pixels.
 *   \param color The color of the Ellipse's fill or outline
 *   \param filled Whether the Ellipse should be filled
 *     (set to true by default).
 */
Ellipse::Ellipse(float x, float y, float z, GLfloat xRadius, GLfloat yRadius, float yaw, float pitch, float roll, ColorFloat color) : ConvexPolygon(x,y,z,(xRadius + yRadius) / 2 + 5 + 1,yaw,pitch,roll) {
    attribMutex.lock();
    myXScale = myXRadius = xRadius;
    myYScale = myYRadius = yRadius;
    myZScale = 1;
    edgesOutlined = false;
    verticesPerColor = ((xRadius + yRadius) / 2 + 6) / 8;
    attribMutex.unlock();
    addVertex(0,0,0,color);
    float delta = 2.0f / (numberOfVertices - 2) * PI;
    for (int i = 0; i < numberOfVertices - 1; ++i) {
        addVertex(cos(i*delta), sin(i*delta), 0, color);
    }
}

/*!
 * \brief Explicitly constructs a new multicolored filled or outlined Ellipse.
 * \details This function draws a Ellipse with the given center, radii, color, and outline color.
 *   \param x The x coordinate of the Ellipse's center.
 *   \param y The y coordinate of the Ellipse's center.
 *   \param xRadius The horizontal radius of the Ellipse in pixels.
 *   \param yRadius The vertical radius of the Ellipse in pixels.
 *   \param color An array of colors for the Ellipse's fill or outline
 *   \param filled Whether the Ellipse should be filled
 *     (set to true by default).
 */
Ellipse::Ellipse(float x, float y, float z, GLfloat xRadius, GLfloat yRadius, float yaw, float pitch, float roll, ColorFloat color[]) : ConvexPolygon(x,y,z,(xRadius + yRadius) / 2 + 5 + 1,yaw,pitch,roll) {
    attribMutex.lock();
    myXScale = myXRadius = xRadius;
    myYScale = myYRadius = yRadius;
    myZScale = 1;
    edgesOutlined = false;
    verticesPerColor = ((xRadius + yRadius) / 2 + 6) / 8;
    attribMutex.unlock();
    addVertex(0,0,0,color[0]);
    float delta = 2.0f / (numberOfVertices - 2) * PI;
    for (int i = 0; i < numberOfVertices - 1; ++i) {
        addVertex(cos(i*delta), sin(i*delta), 0, color[(int) ((float) i / verticesPerColor + 1)]);
    }
}

/**
 * \brief Mutates the horizontal radius of the Ellipse.
 * \param xRadius The Ellipse's new x-radius.
 */
void Ellipse::setXRadius(GLfloat xRadius) {
    if (xRadius <= 0) {
        TsglDebug("Cannot have a Ellipse with x-radius less than or equal to 0.");
        return;
    }
    attribMutex.lock();
    myXRadius = xRadius;
    myXScale = xRadius;
    attribMutex.unlock();
}

/**
 * \brief Mutates the horizontal radius of the Ellipse by the parameter amount.
 * \param delta The amount by which to change the x-radius of the Ellipse.
 */
void Ellipse::changeXRadiusBy(GLfloat delta) {
    if (myXRadius + delta <= 0) {
        TsglDebug("Cannot have a Ellipse with x-radius less than or equal to 0.");
        return;
    }
    attribMutex.lock();
    myXRadius += delta;
    myXScale += delta;
    attribMutex.unlock();
}

/**
 * \brief Mutates the vertical radius of the Ellipse.
 * \param YRadius The Ellipse's new y-radius.
 */
void Ellipse::setYRadius(GLfloat yRadius) {
    if (yRadius <= 0) {
        TsglDebug("Cannot have a Ellipse with y-radius less than or equal to 0.");
        return;
    }
    attribMutex.lock();
    myYRadius = yRadius;
    myYScale = yRadius;
    attribMutex.unlock();
}

/**
 * \brief Mutates the vertical radius of the Ellipse by the parameter amount.
 * \param delta The amount by which to change the y-radius of the Ellipse.
 */
void Ellipse::changeYRadiusBy(GLfloat delta) {
    if (myYRadius + delta <= 0) {
        TsglDebug("Cannot have a Ellipse with y-radius less than or equal to 0.");
        return;
    }
    attribMutex.lock();
    myYRadius += delta;
    myYScale += delta;
    attribMutex.unlock();
}

/**
 * \brief Sets the Ellipse to a new array of colors.
 * \param c An array of the new ColorFloats.
 */
void Ellipse::setColor(ColorFloat c[]) {
    colors[0] = c[0].R;
    colors[1] = c[0].G;
    colors[2] = c[0].B;
    colors[3] = c[0].A;
    int colorIndex;
    for (int i = 1; i < numberOfVertices; ++i) {
        colorIndex = (int) ((float) (i - 1) / verticesPerColor + 1);
        colors[i*4] = c[colorIndex].R;
        colors[i*4 + 1] = c[colorIndex].G;
        colors[i*4 + 2] = c[colorIndex].B;
        colors[i*4 + 3] = c[colorIndex].A;
    }
}


}