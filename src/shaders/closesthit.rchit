#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_scalar_block_layout : require

layout(location = 0) rayPayloadInEXT vec3 hitValue;
hitAttributeEXT vec3 attribs;

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;

#define PI 3.14159265359

// Large buffers containing all mesh data
struct VertexData {
    vec4 position;
    vec4 normal;
	vec2 texcoords;
};

struct Material{
	uint albedoIndex;
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

layout(set = 0, binding = 6, scalar) readonly buffer MaterialBuffer {
	Material materials[];
};

layout(set = 0, binding = 7) uniform sampler2D textures[300];

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

struct RayPayLoad
{
	vec3 hitcolor;
	vec3 hitnormal;
};

layout(location = 1) rayPayloadEXT bool isShadowed;
layout(location = 2) rayPayloadEXT RayPayLoad diffusePayLoad;

float random (vec2 st)
{
	return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

float hash(vec3 p, float seed) {
    return fract(sin(dot(p + seed, vec3(12.9898, 78.233, 45.5432))) * 43758.5453);
}

float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.13);
    p3 += dot(p3, p3.yzx + 3.333);
    return fract((p3.x + p3.y) * p3.z);
}


uint JenkinsHash(uint x)
{
	x += x << 10;
	x ^= x >> 6;
	x += x << 3;
	x ^= x >> 11;
	x += x << 15;

	return x;
}

// need to somehow init the RNG
//uint InitRNG(ivec2 pixel, ivec2 resolution, uint frame)
//{
//	uint rngState = dot(pixel, ivec2(1, resolution.x)) ^ JenkinsHash(frame);
//	return JenkinsHash(rngState);
//}

float UintToFloat(uint x)
{
	return float(0x3f800000 | (x >> 9)) - 1.f;
}

uint XorShift(inout uint rngState)
{
	rngState ^= rngState << 13;
	rngState ^= rngState >> 17;
	rngState ^= rngState << 5;
	return rngState;
}

float Rand(inout uint rngState)
{
	return UintToFloat(XorShift(rngState));
}

vec3 CosHemisphereDirection(vec3 n, vec3 pos, float seed) {
    // Generate two random numbers
    float r1 = hash(pos, seed);
    float r2 = hash(pos + vec3(1.0, 2.0, 3.0), seed); // Offset to get a different value

    // Cosine-weighted sampling
    float theta = 2.0 * 3.14159265359 * r1;  // Azimuthal angle
    float r = sqrt(r2);                      // Radial distance (cosine weighting)
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(1.0 - r2);                // Correctly bias towards normal

    // Convert to world space
    vec3 up = abs(n.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, n));
    vec3 bitangent = cross(n, tangent);

    return tangent * x + bitangent * y + n * z;
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
    float y = cos(phi);
    float z = sin(phi) * sin(theta);

    vec3 dir = vec3(x, y, z);

    // Align with normal (create a basis and transform)
    vec3 up = abs(n.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, n));
    vec3 bitangent = cross(n, tangent);

    return tangent * dir.x + bitangent * dir.y + n * dir.z;
}

float value(vec3 direction, vec3 normal)
{
	float cosTheta = max(0.0f, dot(normal, direction));
	return cosTheta / PI;
}

float ao(vec3 pos, vec3 n, int samples) {
    float a = 0.0;

    for (int i = 0; i < samples; i++) {
        float seed = float(i);
        vec3 dir = CosHemisphereDirection(n, pos, seed);
		isShadowed = true;
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
    }



    return 1.0 - (a / float(samples));
}

#define PI 3.14159265359

uvec2 pcg2d(uvec2 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    return v;
}

float rand(inout uint state)
{
    state = pcg2d(uvec2(state, 0u)).x;
    return float(state) / float(0xffffffffu);
}

vec3 computeLighting(vec3 pos, vec3 n)
{
    uint seed = gl_LaunchIDEXT.x + gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x; // Unique per pixel
    vec3 omega_i = CosHemisphereDirection(n, pos, seed);

    isShadowed = true;
    traceRayEXT(
        topLevelAS,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT,
        0xFF,
        1, // perhaps this means index 1 for when miss shaders begin
        0,
        1, // perhaps this means index 1 within the miss collection we have
        pos,
        0.001,
        omega_i,
        10000,
        1
    );

	float pdfValue = value(omega_i, n);

    if (isShadowed) {
        float contribution = max(dot(n, omega_i), 0.0);
        return vec3(1);
    }

    return vec3(0, 0, 0);
}

vec3 computeDiffuse(vec3 pos, vec3 n)
{
    uint seed = gl_LaunchIDEXT.x;
    vec3 omega_i = CosHemisphereDirection(n, pos, seed);

    // Clear payload to detect if it’s unchanged
    diffusePayLoad.hitcolor = vec3(0.0);
    diffusePayLoad.hitnormal = vec3(0.0);

    traceRayEXT(
        topLevelAS,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
        0xFF,
        1, // Hit group 1 (diffusehit.rchit.spv)
        0,
        1, // Miss index 1 (shadowmiss.rmiss.spv)
        pos,
        0.001,
        omega_i,
        10000,
        2
    );

    // Diffuse BRDF: albedo / PI
    vec3 brdf = diffusePayLoad.hitcolor / PI;
    float L_intensity = 2.0f;
    float cosTheta = max(dot(diffusePayLoad.hitnormal, omega_i), 0.0); // Use input normal for incident angle
    float pdfValue = value(omega_i, diffusePayLoad.hitnormal); // PDF based on input normal

    vec3 outLight = vec3(0.0);
    if (pdfValue > 0.0001) { // Avoid division by zero
        outLight = outLight + (brdf * L_intensity * cosTheta) / pdfValue;
    }

    return outLight; // Final color with albedo
}


// L = normalize(lightpos - worldpos);
vec3 computeReflection(vec3 pos, vec3 dir, vec3 normal, vec3 lightPos, float lightIntensity) {

	isShadowed = true;  // Default to "hit" until proven otherwise
    traceRayEXT(
        topLevelAS,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT,  // Shadow-like flags
        0xFF,                   // Hit all objects
        0,                      // Hit group offset
        0,                      // Geometry offset
        1,                      // Miss shader index (shadow.miss)
        pos,                    // Origin
        0.001,                  // Min distance
        dir,                    // Reflection direction
        10000.0,               // Max distance
        1                       // Payload index for isShadowed
    );

    if (!isShadowed) {
        // Ray missed: reflect environment (e.g., sky)
        return vec3(0.5, 0.7, 1.0);  // Light blue sky
    } else {
        // Ray hit: approximate lighting at the hit point
        vec3 L = normalize(lightPos - pos);  // Light direction (approximation)
        float diff = max(dot(normal, L), 0.0);  // Diffuse term
        return vec3(0.02) + diff * vec3(0.8, 0.8, 0.8) * lightIntensity;  // Ambient + diffuse
    }
}

void main()
{
	const int primitiveID = gl_PrimitiveID;

	const uint meshID = gl_InstanceCustomIndexEXT;
	const uint indexOffset = offsets[meshID].x;
	const uint vertexOffset = offsets[meshID].y;

	Material material = materials[meshID];

	const uint i0 = indices[indexOffset + 3 * primitiveID + 0] + vertexOffset;
	const uint i1 = indices[indexOffset + 3 * primitiveID + 1] + vertexOffset;
	const uint i2 = indices[indexOffset + 3 * primitiveID + 2] + vertexOffset;

	const vec3 v0 = vertices[i0].position.xyz;
	const vec3 v1 = vertices[i1].position.xyz;
	const vec3 v2 = vertices[i2].position.xyz;

	const vec2 uv0 = vertices[i0].texcoords;
	const vec2 uv1 = vertices[i1].texcoords;
	const vec2 uv2 = vertices[i2].texcoords;

	// Interpolating the texture coordinates using barycentric coordinates
	const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
	vec2 interpolatedUV = barycentrics.x * uv0 + barycentrics.y * uv1 + barycentrics.z * uv2;

	// Sample texture using interpolated UVs
	vec3 albedo = texture(textures[material.albedoIndex], interpolatedUV).rgb;

	vec3 pos = v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
	const vec3 worldPos = vec3(gl_ObjectToWorldEXT * vec4(pos, 1.0));

	vec3 objectNormal = normalize(cross(v1 - v0, v2 - v0));

	const vec3 n0 = (vertices[i0].normal.xyz);
	const vec3 n1 = (vertices[i1].normal.xyz);
	const vec3 n2 = (vertices[i2].normal.xyz);

	//objectNormal = (n0 * barycentrics.x + n1 * barycentrics.y + n2 * barycentrics.z);
	vec3 worldNormal = normalize(vec3(objectNormal * gl_WorldToObjectEXT).xyz);

	float lightIntesity = 100.0;
	vec3 lightpos = vec3(300.0, 400.0, 1.0);
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

	float a = ao(worldPos, worldNormal, 4);

	vec3 reflectionDirection = reflect(gl_WorldRayDirectionEXT, worldNormal);
	//vec3 refColor = computeReflection(worldPos, reflectionDirection, worldNormal, lightpos, lightIntesity);
	vec3 diffuse = albedo * diff * att;
	vec3 lighting = computeLighting(worldPos, worldNormal);
    hitValue = computeDiffuse(pos, worldNormal) * albedo; // Grayscale AO
}