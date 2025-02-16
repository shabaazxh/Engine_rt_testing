#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_scalar_block_layout : require

layout(location = 0) rayPayloadInEXT vec3 hitValue;
hitAttributeEXT vec3 attribs;

// Large buffers containing all mesh data
struct VertexData {
    vec3 position;
	vec2 texcoords;
    vec3 normal;
};

layout(set = 0, binding = 3, scalar) readonly buffer VertexBuffer {

    VertexData vertices[];
};


layout(set = 0, binding = 4, scalar) readonly buffer IndexBuffer {
    uint indices[];
};

layout(set = 0, binding = 2) uniform SceneUniform
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

void main()
{
	const int primitiveID = gl_PrimitiveID;

	const uint i0 = indices[3 * primitiveID + 0];
	const uint i1 = indices[3 * primitiveID + 1];
	const uint i2 = indices[3 * primitiveID + 2];

	const vec3 v0 = vertices[i0].position;
	const vec3 v1 = vertices[i1].position;
	const vec3 v2 = vertices[i2].position;

	const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

	vec3 pos = v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
	const vec3 worldPos = vec3(gl_ObjectToWorldEXT * vec4(pos, 1.0));

	vec3 objectNormal = cross(v1 - v0, v2 - v0);

	const vec3 n0 = (vertices[i0].normal);
	const vec3 n1 = (vertices[i1].normal);
	const vec3 n2 = (vertices[i2].normal);

	//objectNormal = (n0 * barycentrics.x + n1 * barycentrics.y + n2 * barycentrics.z);
	vec3 worldNormal = normalize((objectNormal * gl_WorldToObjectEXT).xyz);

	float lightIntesity = 1.0;
	float lightDistance = 10000.0;
	vec3 lightpos = vec3(1.0, 20.0, 1.0);
	vec3 L = normalize(lightpos);
	float diff = max(dot(worldNormal, L), 0.0);
	vec3 diffuse = diff * vec3(1,1,1) * lightIntesity;

	hitValue = (worldNormal);

}