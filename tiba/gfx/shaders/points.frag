#version 330

out vec4 fragColor; // Output color

uniform vec4 color; // Color of the point

void main() {
    float distance = length(2.0 * gl_PointCoord - 1.0);

    // You can adjust the point size here to control the antialiasing quality
    float pointSize = 0.5; // Adjust this value as needed

    // Calculate the alpha (transparency) value based on distance
    float alpha = 1.0 - smoothstep(pointSize, pointSize + 0.02, distance);

    if (alpha <= 0.0) {
        discard; // Discard fragments that are too transparent
    }

    // Set the color of the point. You can replace this with your own color logic.
    fragColor = vec4(color.xyz, color.w*alpha);
}
