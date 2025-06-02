
#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D renderedScene;
layout(set = 0, binding = 1) uniform sampler2D TemporalAccumHistory;

// Combine the deferred lighting pass and the bloom pass to produce the final output image
void main()
{
	vec4 color = clamp(texture(renderedScene, uv), 0.0, 1.0);

	vec4 temp = texture(TemporalAccumHistory, uv).rgba;
	color = temp;

	vec3 ldrColor = color.rgb / (color.rgb + vec3(1.0));
	vec3 gammaCorrectedColor = pow(ldrColor, vec3(1.0 / 2.2));

	gammaCorrectedColor = clamp(gammaCorrectedColor, 0.0, 1.0);
	fragColor = vec4(vec3(gammaCorrectedColor), 1.0);
}