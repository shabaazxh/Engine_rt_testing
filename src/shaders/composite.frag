
#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D renderedScene;
layout(set = 0, binding = 1) uniform sampler2D bloomPass;

// Combine the deferred lighting pass and the bloom pass to produce the final output image
void main()
{
	vec4 lighting = texture(renderedScene, uv);
	vec4 bloom = texture(bloomPass, uv);

	vec3 hdrColor = lighting.rgb;
	vec3 ldrColor = hdrColor / (hdrColor + vec3(1.0));

    float l = texture(renderedScene, uv).x;

	vec3 result = ldrColor;
	vec3 gammaCorrectedColor = pow(result, vec3(1.0 / 2.2));

	fragColor = vec4(vec3(hdrColor), 1.0);
}