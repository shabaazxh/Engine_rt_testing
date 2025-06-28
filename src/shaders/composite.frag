
#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D shading_result;

void main()
{
	vec4 color = clamp(texture(shading_result, uv), 0.0, 1.0);

	vec3 ldrColor = color.rgb / (color.rgb + vec3(1.0));
	vec3 gammaCorrectedColor = pow(ldrColor, vec3(1.0 / 2.2));

	gammaCorrectedColor = clamp(gammaCorrectedColor, 0.0, 1.0);
	fragColor = vec4(vec3(gammaCorrectedColor), 1.0);
}