#version 450

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

layout(push_constant) uniform Push
{
	mat4 ModelMatrix;
	vec4 BaseColourFactor;
	float Metallic;
	float Roughness;
}pc;

layout(location = 0) in vec4 pos;
layout(location = 1) in vec4 normal;
layout(location = 2) in vec2 tex;

layout(location = 0) out vec4 WorldPos;
layout(location = 1) out vec2 uv;
layout(location = 2) out vec4 WorldNormal;
layout(location = 3) out mat3 TBN;

void main()
{
	WorldNormal = normalize(pc.ModelMatrix * vec4(normal.xyz, 0.0));
	uv = tex;
	WorldPos = pc.ModelMatrix * vec4(pos.xyz, 1.0);
	gl_Position = ubo.projection * ubo.view * pc.ModelMatrix * vec4(pos.xyz, 1.0);
}
