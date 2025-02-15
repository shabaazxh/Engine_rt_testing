#version 450

//layout(set = 0, binding = 0) uniform Light
//{
//	vec4 LightPosition;
//	vec4 LightColour;
//	mat4 LightSpaceMatrix;
//}LightUBO;
//
//
struct Light
{
	int Type;
	vec4 LightPosition;
	vec4 LightColour;
	mat4 LightSpaceMatrix;
};

const int NUM_LIGHTS = 16;

layout(set = 0, binding = 0) uniform LightBuffer {
	Light lights[NUM_LIGHTS];
} lightData;

layout(push_constant) uniform Push
{
	mat4 ModelMatrix;
	vec4 BaseColourFactor;
	float Metallic;
	float Roughness;
}pc;

layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 tex;
layout(location = 2) in vec3 normal;
layout(location = 3) in uvec3 compressedTBN;

void main()
{
	gl_Position = lightData.lights[0].LightSpaceMatrix * pc.ModelMatrix * vec4(pos, 1.0);
}
