#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_scalar_block_layout : require

struct Light
{
	int Type;
	vec4 LightPosition;
	vec4 LightColour;
	mat4 LightSpaceMatrix;
};

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

const int NUM_LIGHTS = 26;

layout(set = 0, binding = 8) uniform LightBuffer {
	Light lights[NUM_LIGHTS];
} lightData;

layout(set = 0, binding = 9) uniform RTXSettings
{
    int bounces;
    int frameIndex;
} rtx;


struct RayPayLoad
{
    vec3 pos;
    vec3 dir;
    vec3 colour;
    vec3 hitnormal;
    vec3 hitpos;
    float hit;
};

layout(location = 0) rayPayloadInEXT RayPayLoad rayPayLoad;

hitAttributeEXT vec3 attribs;

layout(location = 1) rayPayloadEXT bool isShadowed;
layout(location = 2) rayPayloadEXT RayPayLoad diffusePayLoad;

float hash(vec3 p, float seed) {
    return fract(sin(dot(p + seed, vec3(12.9898, 78.233, 45.5432))) * 43758.5453);
}

// Reference: https://github.com/NVIDIAGameWorks/RTXGI-DDGI/blob/main/samples/test-harness/shaders/include/Random.hlsl#L42
uint WangHash(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

uint Xorshift(uint seed)
{
    // Xorshift algorithm from George Marsaglia's paper
    seed ^= (seed << 13);
    seed ^= (seed >> 17);
    seed ^= (seed << 5);
    return seed;
}

float GetRandomNumber(inout uint seed)
{
    seed = WangHash(seed);
    return float(Xorshift(seed)) * (1.f / 4294967296.f);
}

vec2 GetRandomHashValue(inout uint seed)
{
    float u = GetRandomNumber(seed) * 2.0 - 1.0;
    float v = GetRandomNumber(seed) * 2.0 - 1.0; // Seed is modified in-place
    return vec2(u, v);
}

vec3 CosHemisphereDirection(vec3 n, vec3 pos, inout uint seed) {
    // Generate two random numbers
    float r1 = GetRandomNumber(seed);
    float r2 = GetRandomNumber(seed); // Offset to get a different value

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

vec3 computeDirectLighting(vec3 worldPos, vec3 worldNormal, vec3 albedo, vec3 sunDirection, float sunIntensity, vec3 sunColor) {
    vec3 L = (sunDirection); // Sun is directional, not a point
    float cosTheta = max(dot(worldNormal, L), 0.0);
    vec3 brdf = albedo / PI; // Diffuse BRDF
    return brdf * sunColor * sunIntensity * cosTheta;
}

float CastShadowRay(vec3 pos, vec3 normal, vec3 lightDir) {

    vec3 L = (lightDir); // Directional light
    isShadowed = true;
    traceRayEXT(
        topLevelAS,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT,
        0xFF,
        0, // Hit group 2 -> shadow hit : sets isShadowed to true
        0,
        1, // Miss index 1 (shadowmiss.rmiss.spv)
        pos + normal * 0.001,
        0.001,
        L,
        10000.0, // Large distance for directional light
        1
    );

    return isShadowed ? 0.0 : 1.0; // 1.0 if not shadowed, 0.0 if shadowed
}
// 0.5, 0.75, 1.0 <- sky light
vec3 computeIndirectLighting(vec3 pos, vec3 n, vec3 albedo, vec3 sunDirection, float sunIntensity, vec3 sunColor) {

    vec3 radiance = vec3(0.0);
    uint seed = uint(gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x) + gl_LaunchIDEXT.x;
    seed *= rtx.frameIndex;

    vec3 throughput = vec3(1.0);

    int bounces = rtx.bounces;

    for (int bounce = 0; bounce < bounces; bounce++)
    {
        // Compute direct lighting at the current hit point
        vec3 directLight = computeDirectLighting(
            pos,
            n,
            albedo,
            sunDirection,
            sunIntensity,
            sunColor
        );

        float visibility = CastShadowRay(pos, n, sunDirection);
        radiance += throughput * directLight * visibility;

//        for(int i = 1; i < NUM_LIGHTS - 1; i++)
//        {
//            vec3 lightDir = normalize(lightData.lights[i].LightPosition.xyz - pos);
//            float dist = length(lightData.lights[i].LightPosition.xyz - pos);
//            float att = 1.0 / (dist * dist); // Tune 0.1 and 0.01
//            vec3 LightColour = lightData.lights[i].LightColour.xyz * att;
//
//            directLight = computeDirectLighting(
//                pos,
//                n,
//                albedo,
//                lightDir,
//                1000.0,
//                LightColour
//            );
//
//            //float visibility = CastShadowRay(pos, n, lightDir);
//            radiance += throughput * directLight;
//        }

        if (bounce == bounces - 1) {
            break;
        }

        // Sample a new direction for indirect bounce
        vec3 omega_i = CosHemisphereDirection(n, pos, seed);

        // Reset payload for the next hit
        diffusePayLoad.pos = vec3(0.0);
        diffusePayLoad.hitpos = vec3(0.0);
        diffusePayLoad.hit = -1.0;
        diffusePayLoad.colour = vec3(0.0);
        diffusePayLoad.hitnormal = vec3(0.0);

        // Trace ray for indirect bounce
        traceRayEXT(
            topLevelAS,
            gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
            0xFF,
            1, // Diffuse hit group
            0,
            0, // Miss shader
            pos,
            0.001,
            omega_i,
            10000.0,
            2
        );

        // Ray hit something
        if (diffusePayLoad.hit > 0.0) {
            vec3 brdf = diffusePayLoad.colour / PI; // Lambertian BRDF
            float cosTheta = max(dot(diffusePayLoad.hitnormal, omega_i), 0.0);
            float pdf = cosTheta / PI; // Cosine-weighted PDF

            if (pdf > 0.001) {
                radiance += throughput *= (brdf * cosTheta) / pdf; // Update throughput for next bounce
            }

            pos = diffusePayLoad.hitpos;
            n = diffusePayLoad.hitnormal;
            albedo = diffusePayLoad.colour;

        } else {
            // ray missed
            radiance += throughput * vec3(0.5, 0.75, 1.0); // Sky color
            break;
        }
    }

    return radiance;
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

    // Compute direct lighting at the first hit
    vec3 sunDirection = normalize(vec3(0.0, 1.0, 0.5)); //.5 for z
    float sunIntensity = 1.0;
    vec3 sunColor = vec3(1.0, 0.95, 0.9);
    float sunAngularSize = 0.8;

    vec3 indirectLight = computeIndirectLighting(worldPos, worldNormal, albedo, sunDirection, sunIntensity, sunColor);
    rayPayLoad.colour = indirectLight;
}