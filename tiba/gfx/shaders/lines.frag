#version 330 core

// https://github.com/Renanse/Ardor3D/blob/7df2f1d0ccae867663ef9b511a443dee029112b0/ardor3d-core/src/main/resources/com/ardor3d/renderer/shader/include/alpha_test.glsl
in VertexData{
  vec4 color;
  // vec2 uv0;
  // noperspective float distance;
} VertexIn;

out vec4 fragColor;

void main() {
    fragColor = VertexIn.color;
}
