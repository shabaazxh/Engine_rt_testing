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

float random (vec2 st)
{
	return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

// Simple hash function to generate a pseudo-random float in [0, 1]
float hash(vec3 p, float seed) {
    return fract(sin(dot(p + seed, vec3(12.9898, 78.233, 45.5432))) * 43758.5453);
}



float UintToFloat(uint x)
{
	return float();
}

// Generate a random direction in a hemisphere around the normal
vec3 randomHemisphereDirection(vec3 n, vec3 pos, float seed) {
    // Use hit position as a base for randomness
    float r1 = hash(pos, seed);
    float r2 = hash(pos + vec3(1.0, 2.0, 3.0), seed); // Offset to get a different value

    // Uniformly sample a direction in a hemisphere
    float theta = 2.0 * 3.14159265359 * r1; // Azimuth angle
    float phi = acos(1.0 - 2.0 * r2);       // Polar angle adjusted for hemisphere

    // Convert to Cartesian coordinates
    float x = sin(phi) * cos(theta);
    float y = sin(phi) * sin(theta);
    float z = cos(phi);

    vec3 dir = vec3(x, y, z);

    // Align with normal (create a basis and transform)
    vec3 up = abs(n.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, n));
    vec3 bitangent = cross(n, tangent);

    return tangent * dir.x + bitangent * dir.y + n * dir.z;
}

// Ambient Occlusion function for closest hit shader
float ao(vec3 pos, vec3 n, int samples) {
    float a = 0.0;

    for (int i = 0; i < samples; i++) {
        // Use sample index as a seed for variation
        float seed = float(i);

        // Generate random direction in hemisphere
        vec3 dir = randomHemisphereDirection(n, pos, seed);
		isShadowed = true;
        // Trace ray (assuming a trace function exists)
        // Replace with your actual ray-tracing call
        traceRayEXT(
			topLevelAS,
			gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT, // Terminate on first hit
			0xFF,
			1, // perhaps this means index 1 for when miss shaders begin
			0,
			1, // perhaps this means index 1 within the miss collection we have
			pos,
			0.001,
			dir,
			10000,
			1
		);

		a += isShadowed == true ? 1.0 :0.0;

        // Accumulate occlusion (0 = occluded, 1 = unoccluded)
    }

    // Average the occlusion and invert for AO
    return 1.0 - (a / float(samples));
}

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

	float lightIntesity = 100.0;
	vec3 lightpos = vec3(300.0, 2000.0, 1.0);
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
			1, // perhaps this means index 1 for when miss shaders begin
			0,
			1, // perhaps this means index 1 within the miss collection we have
			shadowOrigin,
			0.001,
			shadowDirection,
			lightDistance - 0.001,
			1
		);
	}

	// If shadowed, reduce lighting
	if (isShadowed) {
		att = 0.1;
	}

	// check for reflections here?

	vec3 reflectionOrigin = worldPos;
	vec3 reflectionDirection = reflect(gl_WorldRayDirectionEXT, worldNormal);
	float a = ao(worldPos, worldNormal, 4);
//	traceRayEXT(
//		topLevelAS,
//		gl_RayFlagsNoneEXT,
//		0xFF,
//		0,
//		0,
//		0,
//		reflectionOrigin,
//		0.1,
//		reflectionDirection,
//		100000.0,
//		0
//	);

	// Final lighting calculation
	hitValue = vec3(a,a,a);
}