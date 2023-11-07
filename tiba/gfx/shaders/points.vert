#version 330 core

in vec3 position; // Vertex position

uniform float radius;
uniform mat4 mvp; // Model-View-Projection matrix

void main() {
    gl_Position = mvp * vec4(position, 1.0);
    gl_PointSize = radius*2;
}
