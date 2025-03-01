
#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D renderedScene;
layout(set = 0, binding = 1) uniform sampler2D bloomPass;
layout(set = 0, binding = 2) uniform sampler2D[5] historyImages;

// Combine the deferred lighting pass and the bloom pass to produce the final output image
void main()
{
	vec4 color = texture(renderedScene, uv);

	vec4 historyBlend = texture(historyImages[0], uv); // Start with oldest
	for (int i = 1; i < 5; i++) {
		historyBlend = mix(historyBlend, texture(historyImages[i], uv), 0.8); // Blend in each newer frame
	}

	color = mix(color, historyBlend, 0.8);

	vec3 ldrColor = color.rgb / (color.rgb + vec3(1.0));
	vec3 gammaCorrectedColor = pow(ldrColor, vec3(1.0 / 2.2));
	fragColor = vec4(vec3(historyBlend), 1.0);
}