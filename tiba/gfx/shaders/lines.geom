#version 330 core

layout(lines) in;
layout(triangle_strip, max_vertices=8) out;

uniform float   linewidth;
uniform vec2    viewport;
uniform vec2    viewOffset;
uniform float   featherWidth;

// uniform mat4 textureMatrix0;

in VertexData{
  vec4 color;
  // float texV;
} VertexIn[2];

out VertexData{
  vec4 color;
  //vec2 uv0;
  //noperspective float distance;
} VertexOut;

vec3 clipToNDC(vec4 clipPoint)
{
  return clipPoint.xyz / clipPoint.w;
}

vec2 ndcToScreen2D(vec3 ndcPoint, vec2 viewport, vec2 viewOffset)
{
  return (ndcPoint.xy + 1.0) * 0.5 * viewport + viewOffset;
}

vec2 clipToScreen2D(vec4 clipPoint, vec2 viewport, vec2 viewOffset)
{
  return ndcToScreen2D(clipToNDC(clipPoint), viewport, viewOffset);
}

vec3 screen2DToNDC(vec2 screenPoint, vec2 viewport, vec2 viewOffset, float ndcZ) {
  return vec3((2.0 * (screenPoint - viewOffset) / viewport) - 1.0, ndcZ);
}

vec4 ndcToClip(vec3 ndcPoint, float clipW)
{
  return vec4(ndcPoint * clipW, clipW);
}

vec4 screen2DToClip(vec2 screenPoint, vec2 viewport, vec2 viewOffset, float ndcZ, float clipW)
{
  return ndcToClip(screen2DToNDC(screenPoint, viewport, viewOffset, ndcZ), clipW);
}

void emitVert(const vec2 screenPoint, const float ndcZ, const vec4 color, const float distance, const vec2 uv)
{
    //VertexOut.uv0 = uv;
    VertexOut.color = color;
    //VertexOut.distance = distance;
    gl_Position = screen2DToClip(screenPoint, viewport, viewOffset, ndcZ, 1.0);
    EmitVertex();
}

void clampCSW(inout vec4 csA, inout vec4 csB, const float minW)
{
    vec4 csNorm;
    if (csA.w < minW) {
        csNorm = normalize(csB - csA);
        csA += csNorm * ((0.1 - csA.w) / csNorm.w);
    }
    if (csB.w < minW) {
        csNorm = normalize(csA - csB);
        csB += csNorm * ((0.1 - csB.w) / csNorm.w);
    }
}

void main()
{

    vec4 cs0 = gl_in[0].gl_Position;
    vec4 cs1 = gl_in[1].gl_Position;


    // ignore lines that are completely behind us.
    if (cs0.w <= 0 && cs1.w <= 0) return;

    // now clamp points so they end at roughly our view plane if they are behind us.
    // This eliminates texturing issues and oddly flipped vertices.
    clampCSW(cs0, cs1, 0.1);

    // convert the vertices passed to the shader to screen space:
    vec3 ndc0 = clipToNDC(cs0);
    vec3 ndc1 = clipToNDC(cs1);
    vec2 p0 = ndcToScreen2D(ndc0, viewport, viewOffset); // start of current segment
    vec2 p1 = ndcToScreen2D(ndc1, viewport, viewOffset); // end of current segment

    // determine the direction of the segment
    vec2 dir = normalize(p1 - p0);

    // determine the normal of the segment
    vec2 normal = vec2(-dir.y, dir.x);

    // multiply normal by lineWidth to get offset
    // we will add/subtract this to our screen space points and then convert
    // the points back to a modified clip space to get our new triangle strip line.
    vec2 offset = linewidth * 0.5 * normal;

    // come up with our start and end uv coords
    vec2 texVALeft = vec2(0, 0); //vec2(textureMatrix0 * vec4(0.0, VertexIn[0].texV, 1.0, 1.0));
    vec2 texVARight = vec2(0, 0); //vec2(textureMatrix0 * vec4(1.0, VertexIn[0].texV, 1.0, 1.0));
    vec2 texVBLeft = vec2(0, 0); //vec2(textureMatrix0 * vec4(0.0, VertexIn[1].texV, 1.0, 1.0));
    vec2 texVBRight = vec2(0, 0); //vec2(textureMatrix0 * vec4(1.0, VertexIn[1].texV, 1.0, 1.0));

    float distanceA = 0;
    float distanceB = distanceA + length(p1 - p0);

    // generate the triangle strip using two triangles
    // emitVert(p0 + offset, ndc0.z, VertexIn[0].color, distanceA, texVARight);
    // emitVert(p0 - offset, ndc0.z, VertexIn[0].color, distanceA, texVALeft);
    // emitVert(p1 + offset, ndc1.z, VertexIn[1].color, distanceB, texVBRight);
    // emitVert(p1 - offset, ndc1.z, VertexIn[1].color, distanceB, texVBLeft);

    vec2 offsetFeather = (featherWidth + (linewidth * 0.5)) * normal;

    // generate the triangle strip using six triangles
    emitVert(p1 + offsetFeather, ndc1.z, vec4(VertexIn[1].color.xyz, 0.0), distanceB, texVBRight);
    emitVert(p0 + offsetFeather, ndc0.z, vec4(VertexIn[0].color.xyz, 0.0), distanceA, texVARight);
    emitVert(p1 + offset, ndc1.z, VertexIn[1].color, distanceB, texVBRight);
    emitVert(p0 + offset, ndc0.z, VertexIn[0].color, distanceA, texVARight);
    emitVert(p1 - offset, ndc1.z, VertexIn[1].color, distanceB, texVBLeft);
    emitVert(p0 - offset, ndc0.z, VertexIn[0].color, distanceA, texVALeft);
    emitVert(p1 - offsetFeather, ndc1.z, vec4(VertexIn[1].color.xyz, 0.0), distanceB, texVBLeft);
    emitVert(p0 - offsetFeather, ndc0.z, vec4(VertexIn[0].color.xyz, 0.0), distanceA, texVALeft);

    EndPrimitive();
}
