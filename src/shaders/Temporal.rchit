#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_nonuniform_qualifier : enable

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

const int NUM_LIGHTS = 100;

layout(set = 0, binding = 8) uniform LightBuffer {
	Light lights[NUM_LIGHTS];
} lightData;

layout(set = 0, binding = 9) uniform RTXSettings
{
    int bounces;
    int frameIndex;
    int numPastFrames;
    int enable;
    int M;
} rtx;

layout(set = 0, binding = 10) uniform sampler2D InitialCandidates;
layout(set = 0, binding = 11) uniform sampler2D PreviousFrame;
layout(set = 0, binding = 12) uniform sampler2D MotionVectors;
layout(set = 0, binding = 13) uniform sampler2D g_world_positions;
layout(set = 0, binding = 14) uniform sampler2D g_world_normals;


struct RayPayLoad
{
    vec3 pos;
    vec3 dir;
    vec4 colour;
    vec3 hitnormal;
    vec3 hitpos;
    float hit;
};

layout(location = 0) rayPayloadInEXT RayPayLoad rayPayLoad;

hitAttributeEXT vec3 attribs;

layout(location = 1) rayPayloadEXT bool isShadowed;

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

vec2 GetRandomHashValue01(inout uint seed)
{
    float u = GetRandomNumber(seed);
    float v = GetRandomNumber(seed);
    return vec2(u, v);
}


vec2 GetRandomHashValue(inout uint seed)
{
    float u = GetRandomNumber(seed) * 2.0 - 1.0;
    float v = GetRandomNumber(seed) * 2.0 - 1.0; // Seed is modified in-place
    return vec2(u, v);
}


vec3 computeDirectLighting(vec3 worldPos, vec3 worldNormal, vec3 albedo, vec3 lightDirection, float lightIntensity, vec3 lightColour) {
    vec3 L = normalize(lightDirection);
    float cosTheta = max(dot(worldNormal, L), 0.0);
    vec3 brdf = albedo / PI; // Diffuse BRDF - All surfaces assumed Diffuse
    return brdf * lightColour * lightIntensity * cosTheta;
}

vec3 ComputeDiffuseBRDF(vec3 worldPos, vec3 worldNormal, vec3 albedo, vec3 lightDirection, float lightIntensity, vec3 lightColour) {

    vec3 L = normalize(lightDirection);
    float cosTheta = max(dot(worldNormal, L), 0.0);
    vec3 brdf = albedo / PI; // Diffuse BRDF - All surfaces assumed Diffuse
    return brdf * lightColour * lightIntensity * cosTheta;

}

float CastShadowRay(vec3 pos, vec3 normal, vec3 lightDir, float dist) {

    vec3 L = (lightDir); // Directional light
    isShadowed = true;
    traceRayEXT(
        topLevelAS,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT,
        0xFF,
        0,
        0,
        0,
        pos + normal * 0.001,
        0.001,
        L,
        dist, // Large distance for directional light
        1
    );

    return isShadowed ? 0.0 : 1.0; // 1.0 if not shadowed, 0.0 if shadowed
}

struct Reservoir
{
    int index;
    float W_y;
    float totalWeights;
    int M;
};

void update(uint seed, inout Reservoir reservoir, in float xi_weight, int index, int previousM)
{
    reservoir.totalWeights = reservoir.totalWeights + xi_weight;
    float r = GetRandomNumber(seed);
    reservoir.M += previousM; // Paper at the end suggests clamping M for temporal reuse
    if(r < (xi_weight / reservoir.totalWeights))
    {
        reservoir.index = index;
    }
}

// Need to take the current pixel, conver it to UV, then back project it to World , I have the world position from the hit position already
// Then use the previous frames projection * view matrices to reproject it back to UV
// Then sample the previous frame image to get the current pixels reservoir but in the previous frame
// If we have motion vectors, computed we can compute the motion and subtract it from currentUV to get previousUV

// @NOTE: Tip 3.4: Use 1 / M weights if and only if all inputs weights are identically distributed
// If initial candidates have different PDFs, such as when reusing across pixels. When reusing
// across pixels, if you're using different PDFs, the expectation is that nearby pixels might have used
// A different PDF compared to the others thus, MIS is needed to compute a balance heuristic.

// All these initial candidates used a uniform distribution i.e 1 / NUM_LIGHTS

Reservoir combine_reservoirs(vec4 current_pixel_reservoir_data, inout uint seed, vec3 n, vec3 pos, inout vec3 previous_pixel_normal, inout vec3 previous_pixel_position, inout int previous_pixel_reservoir_m)
{
    // Init a reservoir with the current pixel reservoir data
    Reservoir reservoir;;
    reservoir.index = -1;
    reservoir.W_y = 0.0;
    reservoir.M = 0; // Number of candidates used during the initial candidates phase
    reservoir.totalWeights = 0.0;

    // This will hold the two reservoirs, one for the current pixel and one for the previous frame pixel
    Reservoir reservoirs[2];
    for(int i = 0; i < 2; i++) {
        reservoirs[i].index = -1;
        reservoirs[i].W_y = 0.0;
        reservoirs[i].M = 0; // Number of candidates used during the initial candidates phase
        reservoirs[i].totalWeights = 0.0;
    }

    // Get the motion vector for the current pixel
    vec2 motion_vector = texelFetch(MotionVectors, ivec2(gl_LaunchIDEXT.xy), 0).xy;

    // Get the previous frame pixel position by subtracting the motion vector from the current pixel position
    ivec2 current_pixel = ivec2(gl_LaunchIDEXT.xy);
    ivec2 previous_pixel = ivec2(current_pixel - (motion_vector * (ubo.viewportSize))); // motion_vector is difference between UV, we need it in pixels so multiply by viewportsize
    previous_pixel = clamp(previous_pixel, ivec2(0), ivec2(ubo.viewportSize - vec2(1)));

    // get the normal of the previous pixel
    previous_pixel_normal = (texelFetch(g_world_normals, previous_pixel, 0).xyz * 2.0 - 1.0);
    previous_pixel_position = texelFetch(g_world_positions, previous_pixel, 0).xyz;

    reservoirs[0].index = int(current_pixel_reservoir_data.x);
    reservoirs[0].W_y = current_pixel_reservoir_data.y;
    reservoirs[0].M = int(current_pixel_reservoir_data.z);

    reservoirs[1].index = int(texelFetch(PreviousFrame, previous_pixel, 0).x);
    reservoirs[1].W_y   = texelFetch(PreviousFrame, previous_pixel, 0).y;
    reservoirs[1].M = min(int(texelFetch(PreviousFrame, previous_pixel, 0).z), rtx.M); // 20 * reservoirs[0].M
    previous_pixel_reservoir_m = reservoirs[1].M; // Store the previous pixel reservoir M for later use

    for(int i = 0; i < 2; i++) {
        Light L = lightData.lights[reservoirs[i].index];

        float dist = length(L.LightPosition.xyz - pos);
        const vec3 LightDir = normalize(L.LightPosition.xyz - pos);
        const float LightIntensity = 2000.0f * (1.0 / (dist * dist));

        // Evaluate F(x) at the current pixel
        float F_x = max(dot(n, LightDir), 0.0) * LightIntensity; // length(ComputeDiffuseBRDF(pos, n, albedo, LightDir, LightIntensity, L.LightColour.rgb)); // Simplied F(x) only doing diffuse

        // Algorithm 4: Line: 4: p^q(r.y) * r.W * r.M
        float w_i = F_x * reservoirs[i].W_y * reservoirs[i].M;

        // Update the reservoir using current sample data
        update(seed, reservoir, w_i, reservoirs[i].index, reservoirs[i].M);
    }

    return reservoir;
}


vec4 Temporal(vec3 n, vec3 pos)
{
    uint seed = uint(gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x) + gl_LaunchIDEXT.x;
    seed *= rtx.frameIndex;
    vec3 throughput = vec3(1.0);
    vec4 pixelReservoir = texelFetch(InitialCandidates, ivec2(gl_LaunchIDEXT.xy), 0).rgba; // (x = index, y = W_y, z = totalWeights, w = M)


    vec3 previous_pixel_normal = vec3(0);
    vec3 previous_pixel_position = vec3(0);
    int previous_pixel_reservoir_m = -1;
    Reservoir reservoir = combine_reservoirs(pixelReservoir, seed, n, pos, previous_pixel_normal, previous_pixel_position, previous_pixel_reservoir_m);

    // The reservoir should now contain the new updated sample
    // Use the index from the reservoir to fetch the light data
    Light L = lightData.lights[reservoir.index];

    // Compute Z
    int z = 0;
    // Compute F(x) for previous pixel + visibility
    vec3 previous_pixel_lighting_direction = normalize(L.LightPosition.xyz - previous_pixel_position);
    float previous_pixel_light_dist = length(L.LightPosition.xyz - previous_pixel_position);
    float previous_pixel_light_intensity = 2000.0f * (1.0 / (previous_pixel_light_dist * previous_pixel_light_dist));

    // Cast the shadow ray
    float previous_pixel_visibility = CastShadowRay(previous_pixel_position, previous_pixel_normal, previous_pixel_lighting_direction, previous_pixel_light_dist);

    // Compute f(x)
    float previous_pixel_p_hat = max(dot(previous_pixel_normal, previous_pixel_lighting_direction), 0.0) * previous_pixel_light_intensity * previous_pixel_visibility; // F(x) for previous pixel
    if(previous_pixel_p_hat > 0.0)
    {
        // add the previous pixels reservoir M to Z
        z = z + previous_pixel_reservoir_m;
    }

    // Current pixel
    // Compute lighting using this light source
    vec3 LightDir = normalize(L.LightPosition.xyz - pos);
    float dist = length(L.LightPosition.xyz - pos);
    float att = 1.0 / (dist * dist);
    float LightIntensity = 2000.0f * att;
    float F_x = max(dot(n, LightDir), 0.0) * LightIntensity; //length(ComputeDiffuseBRDF(pos, n, albedo, LightDir, intensity, L.LightColour.rgb)); // F(x) = max(dot(n, L), 0.001) * intensity, where n is the normal and L is the light direction

    // Algorithm 4: Line 6: Reservoir s: s.W = 1 / p^q(s.y) * ( 1 / s.M  * s.totalWeights )
    reservoir.W_y = F_x > 0.0 ? (1.0 / F_x) * (1.0 / reservoir.M) * reservoir.totalWeights : 0.0;

    float current_pixel_visbility = CastShadowRay(pos, n, LightDir, dist);
    float current_pixel_p_hat = F_x * current_pixel_visbility;
    if(current_pixel_p_hat > 0.0)
    {
        // add the current pixels reservoir M to Z
        z = z + min(int(pixelReservoir.z), rtx.M);
    }

    float m = (z > 0) ? 1.0 / float(z) : 0.0;


    if(rtx.bounces == 3) {
        // F_x = F_x * current_pixel_visbility;
        reservoir.W_y = F_x > 0.0 ? (1.0 / F_x) * (m * reservoir.totalWeights) : 0.0;
    }

    return vec4(reservoir.index, reservoir.W_y, reservoir.M, 0.0);
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
	vec3 worldPos = vec3(gl_ObjectToWorldEXT * vec4(pos, 1.0));

	vec3 objectNormal = normalize(cross(v1 - v0, v2 - v0));
	vec3 worldNormal = normalize(vec3(objectNormal * gl_WorldToObjectEXT).xyz);

    worldPos = texelFetch(g_world_positions, ivec2(gl_LaunchIDEXT.xy), 0).xyz;
    worldNormal = (texelFetch(g_world_normals, ivec2(gl_LaunchIDEXT.xy), 0).xyz * 2.0 - 1.0);

    //vec3 indirectLight = computeIndirectLightingBEFORENEW(worldPos, worldNormal, albedo, sunIntensity);
    // vec3 DirectLight = RISReservoirSampling(worldPos, worldNormal, albedo);
    //vec3 DirectLight = NaiveDirectLighting(worldPos, worldNormal, albedo);
//    if(rtx.bounces == 1)
//    {
//        rayPayLoad.colour = vec3(Verify(worldNormal, worldPos, albedo));
//    } else {
//        vec3 radiance = vec3(0.3, 0.5, 0.9);
//        rayPayLoad.colour = vec3(Temporal(worldNormal, worldPos, albedo));
//    }

    rayPayLoad.colour = vec4(Temporal(worldNormal, worldPos));
}