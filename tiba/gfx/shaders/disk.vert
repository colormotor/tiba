#version 330

uniform vec2 resolution;
uniform vec2 viewport;
uniform mat4 mvp;
uniform mat4 ortho;

in vec3 position;
uniform float radius;
out vec2 v_center;
out float v_radius;

void main()
{
    v_radius = radius;

    vec4 pos = mvp*vec4(position, 1.0);
    gl_Position = ortho*vec4((pos.xy / pos.w)*viewport, 0.0, 1.0);
    v_center = vec4((pos.xy / pos.w)*viewport, 0.0, 1.0).xy;
}
