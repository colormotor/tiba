#version 330

uniform float antialias;
uniform float linewidth;
uniform float miter_limit;
uniform vec2 viewport;
uniform mat4 mvp;
in vec3 position;
out float v_antialias;
out float v_linewidth;
out float v_miter_limit;
void main()
{
    v_antialias = antialias;
    v_linewidth = linewidth;
    v_miter_limit = miter_limit;
    vec4 pos = mvp*vec4(position, 1.0);
    gl_Position = vec4((pos.xy / pos.w)*viewport, 0.0, 1.0);
}
