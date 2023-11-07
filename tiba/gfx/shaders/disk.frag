#version 330

in float v_radius;
in vec2 v_center;
out vec4 fragColor;
uniform vec4  color;

void main()
{
    vec2 p = gl_FragCoord.xy - v_center;
    float a = 1.0;
    float d = length(p) - v_radius + 1.0;
    if(d > 0.0) a = exp(-d*d);
    fragColor = vec4(color.xyz, a*0.5); // color.w*a);
}
