#version 330 core

out VertexData{
  vec4 color;
  // float texV;
} VertexOut;

uniform mat4 mvp;
uniform vec4 color;

in vec3 position;

void main() {
    // VertexOut.texV = 0.0;
    VertexOut.color = color;
    gl_Position = mvp * vec4(position, 1.0);
}
