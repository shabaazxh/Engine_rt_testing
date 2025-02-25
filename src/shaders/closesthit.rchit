#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_scalar_block_layout : require

struct RayPayLoad
{
    vec3 colour;
    vec3 normal;
    float hit;
};

layout(location = 0) rayPayloadInEXT RayPayLoad rayPayLoad;

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

layout(location = 1) rayPayloadEXT bool isShadowed;
layout(location = 2) rayPayloadEXT RayPayLoad diffusePayLoad;

float hash(vec3 p, float seed) {
    return fract(sin(dot(p + seed, vec3(12.9898, 78.233, 45.5432))) * 43758.5453);
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

float value(vec3 direction, vec3 normal)
{
	float cosTheta = max(0.0f, dot(normal, direction));
	return cosTheta / PI;
}
// 0.5, 0.75, 1.0 <- sky light
vec3 computeIndirectLighting(vec3 pos, vec3 n, float sunIntensity)
{
    vec3 outLight = vec3(0.0);
    uint seed = gl_LaunchIDEXT.x;
    vec3 omega_i = CosHemisphereDirection(n, pos, seed);

    // Clear payload to detect if it’s unchanged
    diffusePayLoad.colour = vec3(0.0);
    diffusePayLoad.normal = vec3(0.0);
    diffusePayLoad.hit = -1.0;


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

    if(diffusePayLoad.hit > 0.0) {
        // Diffuse BRDF: albedo / PI
        vec3 brdf = diffusePayLoad.colour / PI;
        float L_intensity = sunIntensity;
        float cosTheta = max(dot(diffusePayLoad.normal, omega_i), 0.0); // Use input normal for incident angle
        float pdfValue = value(omega_i, diffusePayLoad.normal); // PDF based on input normal

        if (pdfValue > 0.0001) { // Avoid division by zero
            outLight = outLight + (brdf * L_intensity * cosTheta) / pdfValue;
        }
    } else
    {
        outLight += vec3(0.5, 0.75, 1.0);
    }

    return outLight; // Final color with albedo
}

vec3 computeDirectLighting(vec3 worldPos, vec3 worldNormal, vec3 albedo, vec3 sunDirection, float sunIntensity, vec3 sunColor) {
    // Direction to the light
    vec3 L = sunDirection;

    // Shadow ray
    isShadowed = true;
    traceRayEXT(
        topLevelAS,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT,
        0xFF,
        1, // Hit group 1
        0,
        1, // Miss index 1 (shadowmiss.rmiss.spv)
        worldPos + worldNormal * 0.001, // Offset to avoid self-intersection
        0.001,
        L,
        10000.0, // Far distance for directional light
        1
    );

    // If not shadowed, compute direct lighting
    if (!isShadowed) {
        float cosTheta = max(dot(worldNormal, L), 0.0);
        vec3 brdf = albedo / PI; // Diffuse BRDF
        return brdf * sunColor * sunIntensity * cosTheta;
    }
    return vec3(0.0); // In shadow, no direct contribution
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

	const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
	vec2 interpolatedUV = barycentrics.x * uv0 + barycentrics.y * uv1 + barycentrics.z * uv2;

	vec3 albedo = texture(textures[material.albedoIndex], interpolatedUV).rgb;

	vec3 pos = v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
	const vec3 worldPos = vec3(gl_ObjectToWorldEXT * vec4(pos, 1.0));

	vec3 objectNormal = normalize(cross(v1 - v0, v2 - v0));
	vec3 worldNormal = normalize(vec3(objectNormal * gl_WorldToObjectEXT).xyz);

    vec3 sunDirection = normalize(vec3(0.5, 20, 6.0));
    float sunIntensity = 1.0;
    vec3 sunColor = vec3(1.0, 0.95, 0.9);
    float sunAngularSize = 0.8;

    // Compute direct lighting at the first hit
    vec3 directLight = computeDirectLighting(worldPos, worldNormal, albedo, sunDirection, sunIntensity, sunColor);
    vec3 indirectLight = computeIndirectLighting(worldPos, worldNormal, sunIntensity);
    rayPayLoad.colour = directLight + indirectLight * albedo;
}