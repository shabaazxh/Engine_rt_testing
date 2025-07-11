#version 450

layout(location = 0) in vec4 WorldPos;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec4 WorldNormal;

layout(location = 0) out vec4 g_albedo;
layout(location = 1) out vec4 g_normal;
layout(location = 2) out vec4 g_world_position;
layout(location = 3) out vec2 g_metallicRoughness;

layout(set = 0, binding = 0) uniform SceneUniform
{
	mat4 model;
	mat4 view;
	mat4 projection;
    vec4 cameraPosition;
    vec2 viewportSize;
	float fov;
	float nearPlane;
	float farPlane;
} ubo;

struct Light
{
	int Type;
	vec4 LightPosition;
	vec4 LightColour;
	mat4 LightSpaceMatrix;
};

const int NUM_LIGHTS = 26;

layout(set = 0, binding = 1) uniform LightBuffer {
	Light lights[NUM_LIGHTS];
} lightData;

layout(push_constant) uniform Push
{
	mat4 ModelMatrix;
	vec4 BaseColourFactor;
	float Metallic;
	float Roughness;
}pc;

layout(set = 0, binding = 2) uniform sampler2DShadow shadowMap;

layout(set = 1, binding = 0) uniform sampler2D albedoTexture;
layout(set = 1, binding = 1) uniform sampler2D metallicRoughness;

void main()
{
	vec4 color = texture(albedoTexture, uv);
    vec3 world_normal = normalize(WorldNormal).xyz;
	float roughness = texture(metallicRoughness, uv).b;
	float metallic = texture(metallicRoughness, uv).g;

	g_albedo = color;
	g_normal = vec4(world_normal * 0.5 + 0.5, 0.0); // convert to 0-1 range for packing
	g_world_position = WorldPos;
	g_metallicRoughness = vec2(metallic, roughness);
}
