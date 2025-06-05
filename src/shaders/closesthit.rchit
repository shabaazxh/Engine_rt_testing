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
layout(set = 0, binding = 10, rgba32f) uniform image2D WorldPositionImage;
layout(set = 0, binding = 12, rgba32f) uniform image2D ReservoirsImage;
layout(set = 0, binding = 13, rgba32f) uniform image2D MCandidatesImage;

struct Res
{
    int Y;
    float Y_weight;
};

layout(std430, binding = 11) buffer ReservoirBuffer {
    Res reservoirs[];
}ReservoirStorageBuffer;

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

const int NUM_LIGHTS = 51;

layout(set = 0, binding = 8) uniform LightBuffer {
	Light lights[NUM_LIGHTS];
} lightData;

layout(set = 0, binding = 9) uniform RTXSettings
{
    int bounces;
    int frameIndex;
    int numPastFrames;
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
    float v = GetRandomNumber(seed) * 2.0 - 1.0; // @Note: Seed is modified in place
    return vec2(u, v);
}

vec3 FresnelShlick(float cosTheta, float metalness)
{
    vec3 F0 = vec3(0.04);
    F0 = mix(vec3(0.04), F0, metalness);
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

vec3 computeDirectLighting(vec3 worldPos, vec3 worldNormal, vec3 albedo, vec3 lightDirection, float lightIntensity, vec3 lightColour) {
    vec3 L = normalize(lightDirection);
    float cosTheta = max(dot(worldNormal, L), 0.0);
    vec3 brdf = albedo / PI; // Diffuse BRDF - All surfaces assumed Diffuse
    return brdf * lightColour * lightIntensity * cosTheta;
}


vec3 DiffuseBRDF(vec3 normal, vec3 position, vec3 albedo, vec3 LightDirection, vec3 LightColour, float LightIntensity)
{
    float metallic = 0.1;
    vec3 L = normalize(LightDirection);
    vec3 N = normal;
    vec3 V = normalize(ubo.cameraPosition.xyz - position);
    vec3 H = normalize(V + L);

    vec3 FresnelReflectance = FresnelShlick(max(dot(H, V), 0.0), metallic);
    vec3 kD = vec3(1.0) - FresnelReflectance; // Diffuse component
    kD *= 1.0 - metallic;

    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = (kD * albedo / PI) * LightColour * NdotL;

    return diffuse;
}

float CastShadowRay(vec3 pos, vec3 normal, vec3 lightDir, float dist) {

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
        dist, // Large distance for directional light
        1
    );

    return isShadowed ? 0.0 : 1.0; // 1.0 if not shadowed, 0.0 if shadowed
}

#define CANDIDATE_MAX 32

// @NOTE: Reservoir can also store the PDF. This is useful if we have multiple PDF which you draw samples from.
// This is needed for MIS which will need to be computed if using multiple PDF in both spatial and temporal
struct Reservoir
{
    int index;
    float W_y;
    float totalWeights;
    int M;
};

// This is Weighted Reservoir Sampling with RIS
void update(inout uint seed, inout Reservoir reservoir, in float xi_weight, int index)
{
    reservoir.totalWeights = reservoir.totalWeights + xi_weight;
    float r = GetRandomNumber(seed);
    reservoir.M = reservoir.M + 1;
    if(r < (xi_weight / reservoir.totalWeights))
    {
        reservoir.index = index;
    }
}

void RISReservoir(inout Reservoir reservoir, inout uint seed, vec3 pos, vec3 n, vec3 albedo)
{
    const float rcpUniformDistributionWeight = float(NUM_LIGHTS); // PDF of uniform distribution = 1 / total number of lights. Reciporal of that PDF is the light count e.g. 1 / 10 = 0.1 -> rcp = 1 / (1 / 10) = 10.0
    const float rcpM = 1.0 / float(CANDIDATE_MAX);

    // Picking any light direction has a uniform distribution
    for (int i = 0; i < CANDIDATE_MAX; i++) {

        // Pick a random light from all lights
        int randomLightIndex = int(GetRandomNumber(seed) * float(NUM_LIGHTS));
        Light light = lightData.lights[randomLightIndex];

        // Get the properties of this light
        float dist = length(light.LightPosition.xyz - pos);
        vec3 light_dir = normalize(light.LightPosition.xyz - pos);
        float LightIntensity = 100.0f * (1.0 / (dist * dist));

        // Compute RIS weight for this candidate light
        float F_x = max(dot(n, light_dir), 0.001) * LightIntensity; // Simplied F(x) for weighting. Not sure if need to compute entir BRDF * cosine * ....?  // The target function F(x) that PDF(X) approximates better with more candidates. Using lambert cosine term but this can be other importance sampling methods

        // This is p^q(x_i) / p(x_i) where p^q(x_i) is the target function F_x and p(x_i) is the PDF of the uniform distribution which is 1 / NUM_LIGHTS. So we can compute the weight as F_x * rcpUniformDistributionWeight = F_x * (1 / NUM_LIGHTS) = F_x / NUM_LIGHTS
        float xi_weight = rcpM * F_x * rcpUniformDistributionWeight; // Move 1.0 / M to here when computing weight as suggested
        update(seed, reservoir, xi_weight, randomLightIndex);
    }
}


/*

Weighting in RIS works slightly differently to tranditional. While weighting is used, its not a PDF.
Instead we have an unbiased contribution weight called W_x which replaces 1/P(X).
A single sample X can have many valid W_x depending on the initial candidates samples.
This is because the initial candidates are chosen at random and for each set of them, any sample will
have a different weighing relative to them.
This means W_x is not a determiistic function of X, its a random variable. Since they're unbiased,
1/P(X) can be replaced by W_x.

*/

vec3 RISReservoirSampling(vec3 pos, vec3 n, vec3 albedo)
{
    uint seed = uint(gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x) + gl_LaunchIDEXT.x;
    seed *= rtx.frameIndex;

    vec3 radiance = vec3(0.0);
    vec3 throughput = vec3(1.0);

    Reservoir reservoir;
    reservoir.index = -1;
    reservoir.W_y = 0.0;
    reservoir.totalWeights = 0.0;
    reservoir.M = 0;

    // Compute the weights of the candidates from the original distribution
    RISReservoir(reservoir, seed, pos, n, albedo);

    bool isValidIndex = reservoir.index > -1;

    if(isValidIndex) {
        // The selected light
        int light_index = reservoir.index;
        Light LightSource = lightData.lights[light_index];
        float dist = length(LightSource.LightPosition.xyz - pos);
        vec3 light_dir = normalize(LightSource.LightPosition.xyz - pos);
        float LightIntensity = 100.0f * (1.0 / (dist * dist));

        // Compute the light weight to prevent bias
        // W_x = (sum(w_i) / M) / pdf(x)
        // Written as: 1 / pdf(x) * (1 / m * sum(w_i)), but remember 1 / pdf(x) and 1 / m is the same as dividing by them since 1 / x is rcp
        float Fx = max(dot(n, light_dir), 0.001) * LightIntensity;
        float target_function = 1.0 / Fx; //  float(NUM_LIGHTS);

        // This is debug to ensure its valid, remove eventually -> this has an issue atm @TODO
        if (isinf(target_function))
            return vec3(1.0, 0.0, 1.0);

        // Evaluate the unbiased constribuion weight W_x
        // We moved rcpM = 1 / float(CANDIDATE_MAX) to func RISReservoir which is computing weight for each candidate as suggested by paper
        reservoir.W_y = target_function * (reservoir.totalWeights);

        // Store the current select sample Y, number of candidates M, and probabilistic weight W_y
        imageStore(ReservoirsImage, ivec2(gl_LaunchIDEXT.xy), vec4(reservoir.index, reservoir.W_y, reservoir.M, 0.0));

        float Visibility = CastShadowRay(pos, n, light_dir, dist - 0.001);
        vec3 directLighting = computeDirectLighting(pos, n, albedo, light_dir, LightIntensity, LightSource.LightColour.rgb) * reservoir.W_y * Visibility;
        radiance += throughput * directLighting;
    }

    return radiance;
}


vec3 NaiveDirectLighting(vec3 pos, vec3 n, vec3 albedo)
{
    vec3 radiance = vec3(0.0);
    vec3 throughput = vec3(1.0);

    for (int i = 0; i < NUM_LIGHTS; i++) {

        Light light = lightData.lights[i];

        vec3 lightDir;
        vec3 LightColour;
        float intensity = 0.0f;

        // We have no directional light source so we're not checking directly for directional or other
        lightDir = normalize(light.LightPosition.xyz - pos);
        float dist = length(light.LightPosition.xyz - pos);
        float att = 1.0 / (dist * dist);
        LightColour = light.LightColour.xyz;
        intensity = 100.0f * att;

        float visibility = CastShadowRay(pos, n, lightDir, length(light.LightPosition.xyz - pos) - 0.001);
        vec3 directLighting = computeDirectLighting(pos, n, albedo, lightDir, intensity, LightColour.rgb) * visibility;
        radiance += throughput * directLighting;
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

    imageStore(WorldPositionImage, ivec2(gl_LaunchIDEXT.xy), vec4(worldPos, 0.0));

	vec3 objectNormal = normalize(cross(v1 - v0, v2 - v0));
	vec3 worldNormal = normalize(vec3(objectNormal * gl_WorldToObjectEXT).xyz);

    // Compute direct lighting at the first hit
    vec3 sunDirection = normalize(vec3(0.0f, 1.0f, 0.5)); //.5 for z // vec3(0.0, 1.0, 0.35)
    float sunIntensity = 20.0;
    vec3 sunColor = vec3(1.0, 0.95, 0.9);
    float sunAngularSize = 0.8;

    //vec3 indirectLight = computeIndirectLightingBEFORENEW(worldPos, worldNormal, albedo, sunIntensity);
    vec3 DirectLight = RISReservoirSampling(worldPos, worldNormal, albedo);
    //vec3 DirectLight = NaiveDirectLighting(worldPos, worldNormal, albedo);
    rayPayLoad.colour = DirectLight;
}