#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_scalar_block_layout : require

layout(location = 0) rayPayloadInEXT vec3 hitValue;
hitAttributeEXT vec3 attribs;

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;

// Large buffers containing all mesh data
struct VertexData {
    vec4 position;
    vec4 normal;
	vec2 texcoords;
};

layout(set = 0, binding = 3, scalar) readonly buffer VertexBuffer {

    VertexData vertices[];
};


layout(set = 0, binding = 4, scalar) readonly buffer IndexBuffer {
    uint indices[];
};

// Offsets buffer storing index and vertex offsets for each mesh instance
layout(set = 0, binding = 5, scalar) readonly buffer OffsetBuffer {
    uvec2 offsets[];
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


// Shadow ray payload (only needs a boolean to check occlusion)
layout(location = 1) rayPayloadEXT bool isShadowed;


void main()
{
	const int primitiveID = gl_PrimitiveID;

	const uint meshID = gl_InstanceCustomIndexEXT;
	const uint indexOffset = offsets[meshID].x;
	const uint vertexOffset = offsets[meshID].y;

	const uint i0 = indices[indexOffset + 3 * primitiveID + 0] + vertexOffset;
	const uint i1 = indices[indexOffset + 3 * primitiveID + 1] + vertexOffset;
	const uint i2 = indices[indexOffset + 3 * primitiveID + 2] + vertexOffset;

	const vec3 v0 = vertices[i0].position.xyz;
	const vec3 v1 = vertices[i1].position.xyz;
	const vec3 v2 = vertices[i2].position.xyz;

	const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

	vec3 pos = v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
	const vec3 worldPos = vec3(gl_ObjectToWorldEXT * vec4(pos, 1.0));

	vec3 objectNormal = normalize(cross(v1 - v0, v2 - v0));

	const vec3 n0 = (vertices[i0].normal.xyz);
	const vec3 n1 = (vertices[i1].normal.xyz);
	const vec3 n2 = (vertices[i2].normal.xyz);

	//objectNormal = (n0 * barycentrics.x + n1 * barycentrics.y + n2 * barycentrics.z);
	vec3 worldNormal = normalize(vec3(objectNormal * gl_WorldToObjectEXT).xyz);
	//worldNormal = -worldNormal;

	float lightIntesity = 1.0;
	vec3 lightpos = vec3(1.0, 2000.0, 1.0);
	vec3 L = normalize(lightpos - worldPos);  // Corrected light direction
	float diff = max(dot(worldNormal, L), 0.0);

	// Shadow Ray Setup
	vec3 shadowOrigin = worldPos + worldNormal * 0.001;  // Offset to prevent self-shadowing
	vec3 shadowDirection = normalize(lightpos - shadowOrigin);
	float lightDistance = length(lightpos - shadowOrigin);

	float att = 1.0;
	if(dot(worldNormal, L) > 0) {

		// trace shadow ray
		isShadowed = true;
		traceRayEXT(
			topLevelAS,
			gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT, // Terminate on first hit
			0xFF,
			1,
			0,
			1,
			shadowOrigin,
			0.001,
			shadowDirection,
			lightDistance - 0.001,
			1
		);
	}

	// If shadowed, reduce lighting
	if (isShadowed) {
		att = 0.3;
	}

	// Final lighting calculation
	hitValue = (lightIntesity * (att) * (diff * vec3(1,1,1)));
}