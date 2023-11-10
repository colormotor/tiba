#version 330

layout(lines_adjacency) in; // 4 points at the time from vertex shader
layout(triangle_strip, max_vertices = 4) out; // Outputs a triangle strip with 4 vertices

uniform mat4 projection;
uniform float antialias;
uniform float linewidth;
uniform float miter_limit;
uniform vec2 viewport;

//in float v_antialias[];
//in float v_linewidth[];
//in float v_miter_limit[];
out vec2 v_caps;
out float v_length;
out vec2 v_texcoord;
out vec2 v_bevel_distance;

float compute_u(vec2 p0, vec2 p1, vec2 p)
{
    // Projection p' of p such that p' = p0 + u*(p1-p0)
    // Then  u *= length(p1-p0)
    vec2 v = p1 - p0;
    float l = length(v);
    return ((p.x-p0.x)*v.x + (p.y-p0.y)*v.y) / l;
}

float line_distance(vec2 p0, vec2 p1, vec2 p)
{
    // Projection p' of p such that p' = p0 + u*(p1-p0)
    vec2 v = p1 - p0;
    float l2 = v.x*v.x + v.y*v.y;
    float u = ((p.x-p0.x)*v.x + (p.y-p0.y)*v.y) / l2;
    // h is the projection of p on (p0,p1)
    vec2 h = p0 + u*v;
    return length(p-h);
}

vec4 project(vec2 p, float z) {
    return vec4((projection*vec4(p.xy, 0, 1.0)).xy, z+1e-5, 1.0);
    vec4 pp = projection*vec4(p.xy, 0, 1.0);
    return vec4(2*(p.xy/viewport), z+1e-5, 1.0);
}

void main(void)
{
    //float antialias = v_antialias[0];
    //float linewidth = v_linewidth[0];
    //float miter_limit = v_miter_limit[0];
    // Get the four vertices passed to the shader
    vec3 p0 = gl_in[0].gl_Position.xyz; // start of previous segment
    vec3 p1 = gl_in[1].gl_Position.xyz; // end of previous segment, start of current segment
    vec3 p2 = gl_in[2].gl_Position.xyz; // end of current segment, start of next segment
    vec3 p3 = gl_in[3].gl_Position.xyz; // end of next segment

    /* perform naive culling */
    // vec2 area = viewport * 4;
    // if( p1.x < -area.x || p1.x > area.x ) return;
    // if( p1.y < -area.y || p1.y > area.y ) return;
    // if( p2.x < -area.x || p2.x > area.x ) return;
    // if( p2.y < -area.y || p2.y > area.y ) return;

    // Determine the direction of each of the 3 segments (previous, current, next)
    vec2 v0 = normalize(p1.xy - p0.xy);
    vec2 v1 = normalize(p2.xy - p1.xy);
    vec2 v2 = normalize(p3.xy - p2.xy);
    // Determine the normal of each of the 3 segments (previous, current, next)
    vec2 n0 = vec2(-v0.y, v0.x);
    vec2 n1 = vec2(-v1.y, v1.x);
    vec2 n2 = vec2(-v2.y, v2.x);
    // Determine miter lines by averaging the normals of the 2 segments
    vec2 miter_a = normalize(n0 + n1); // miter at start of current segment
    vec2 miter_b = normalize(n1 + n2); // miter at end of current segment
    // Determine the length of the miter by projecting it onto normal
    vec2 p,v;
    float d;
    float w = linewidth/2.0 + antialias;
    v_length = length(p2-p1);
    float length_a = w / dot(miter_a, n1);
    float length_b = w / dot(miter_b, n1);
    float m = miter_limit*linewidth/2.0;
    // Angle between prev and current segment (sign only)
    float d0 = -sign(v0.x*v1.y - v0.y*v1.x);
    // Angle between current and next segment (sign only)
    float d1 = -sign(v1.x*v2.y - v1.y*v2.x);
    // Generate the triangle strip
    // First vertex
    // ------------------------------------------------------------------------
    // Cap at start
    if( p0 == p1 ) {
        p = p1.xy - w*v1 + w*n1;
        v_texcoord = vec2(-w, +w);
        v_caps.x = v_texcoord.x;
    // Regular join
    } else {
        p = p1.xy + length_a * miter_a;
        v_texcoord = vec2(compute_u(p1.xy, p2.xy, p), +w);
        v_caps.x = 1.0;
    }
    if( p2.xy == p3.xy ) v_caps.y = v_texcoord.x;
    else           v_caps.y = 1.0;
    gl_Position = project(p, p1.z); //projection*vec4(p, p1.z, 1.0);
    v_bevel_distance.x = +d0*line_distance(p1.xy+d0*n0*w, p1.xy+d0*n1*w, p);
    v_bevel_distance.y =    -line_distance(p2.xy+d1*n1*w, p2.xy+d1*n2*w, p);
    EmitVertex();
    // Second vertex
    // ------------------------------------------------------------------------
    // Cap at start
    if( p0 == p1 ) {
        p = p1.xy - w*v1 - w*n1;
        v_texcoord = vec2(-w, -w);
        v_caps.x = v_texcoord.x;
    // Regular join
    } else {
        p = p1.xy - length_a * miter_a;
        v_texcoord = vec2(compute_u(p1.xy, p2.xy, p), -w);
        v_caps.x = 1.0;
    }
    if( p2.xy == p3.xy ) v_caps.y = v_texcoord.x;
    else           v_caps.y = 1.0;
    gl_Position = project(p, p2.z); //projection*vec4(p, p1.z, 1.0);
    v_bevel_distance.x = -d0*line_distance(p1.xy+d0*n0*w, p1.xy+d0*n1*w, p);
    v_bevel_distance.y =    -line_distance(p2.xy+d1*n1*w, p2.xy+d1*n2*w, p);
    EmitVertex();
    // Third vertex
    // ------------------------------------------------------------------------
    // Cap at end
    if( p2.xy == p3.xy ) {
        p = p2.xy + w*v1 + w*n1;
        v_texcoord = vec2(v_length+w, +w);
        v_caps.y = v_texcoord.x;
    // Regular join
    } else {
        p = p2.xy + length_b * miter_b;
        v_texcoord = vec2(compute_u(p1.xy, p2.xy, p), +w);
        v_caps.y = 1.0;
    }
    if( p0.xy == p1.xy ) v_caps.x = v_texcoord.x;
    else           v_caps.x = 1.0;
    gl_Position = project(p, p1.z);
    v_bevel_distance.x =    -line_distance(p1.xy+d0*n0*w, p1.xy+d0*n1*w, p);
    v_bevel_distance.y = +d1*line_distance(p2.xy+d1*n1*w, p2.xy+d1*n2*w, p);
    EmitVertex();
    // Fourth vertex
    // ------------------------------------------------------------------------
    // Cap at end
    if( p2.xy== p3.xy ) {
        p = p2.xy + w*v1 - w*n1;
        v_texcoord = vec2(v_length+w, -w);
        v_caps.y = v_texcoord.x;
    // Regular join
    } else {
        p = p2.xy - length_b * miter_b;
        v_texcoord = vec2(compute_u(p1.xy,p2.xy,p), -w);
        v_caps.y = 1.0;
    }
    if( p0.xy == p1.xy ) v_caps.x = v_texcoord.x;
    else           v_caps.x = 1.0;
    gl_Position = project(p, p2.z);
    v_bevel_distance.x =    -line_distance(p1.xy+d0*n0*w, p1.xy+d0*n1*w, p);
    v_bevel_distance.y = -d1*line_distance(p2.xy+d1*n1*w, p2.xy+d1*n2*w, p);
    EmitVertex();
    EndPrimitive();
}
