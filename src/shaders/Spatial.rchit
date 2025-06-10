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

layout(set = 0, binding = 10) uniform sampler2D InitialCandidatesImage;
layout(set = 0, binding = 11, rgba32f) uniform image2D SpatialReservoirStore;
layout(set = 0, binding = 12) uniform sampler2D TemporalReuseReservoirs;

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

#define CANDIDATE_MAX 8

struct Candidate
{
    Light light;
    float Fx;
    float weight;
    vec3 lightDir;
    float intensity;
};

struct Reservoir
{
    Candidate Y;
    int index;
    float W_y;
    float totalWeights;
    int M;
};

void update(uint seed, inout Reservoir reservoir, in Candidate xi, in float xi_weight, in float f_x, int index);

//void update(uint seed, inout Reservoir reservoir, in Candidate xi, in float xi_weight, int index)
//{
//    reservoir.totalWeights = reservoir.totalWeights + xi_weight;
//    float r = GetRandomNumber(seed);
//    if(r < (xi_weight / reservoir.totalWeights))
//    {
//        reservoir.Y = xi;
//        reservoir.index = index;
//    }
//}

void RISReservoir(inout Reservoir reservoir, uint seed, vec3 pos, vec3 n, inout Candidate candidates[CANDIDATE_MAX], vec3 albedo)
{
    const float rcpUniformDistributionWeight = float(NUM_LIGHTS); // PDF of uniform distribution = 1 / total number of lights. Reciporal of that PDF is the light count e.g. 1 / 10 = 0.1 -> rcp = 1 / (1 / 10) = 10.0
    const float rcpM = 1.0 / float(CANDIDATE_MAX); // This is MIS weight. We are using uniform sampling so its 1 / CANDIDATE_MAX. This is the same as 1 / M in the paper, where M is the number of candidates (8 in this case).

    // Picking any light direction has a uniform distribution
    for (int i = 0; i < CANDIDATE_MAX; i++) {

        // Pick a random light from all lights
        int randomLightIndex = int(GetRandomNumber(seed) * float(NUM_LIGHTS));
        candidates[i].light = lightData.lights[randomLightIndex];

        // Get the properties of this light
        float dist = length(candidates[i].light.LightPosition.xyz - pos);
        float att = 1.0 / (dist * dist);
        candidates[i].lightDir = normalize(candidates[i].light.LightPosition.xyz - pos);
        candidates[i].intensity = 2000.0f * att;

        // Compute RIS weight for this candidate light
        float F_x = max(dot(n, candidates[i].lightDir), 0.001) * candidates[i].intensity; // Simplied F(x) for weighting. Not sure if need to compute entir BRDF * cosine * ....?
        candidates[i].Fx = F_x; // The target function F(x) that PDF(X) approximates better with more candidates. Using lambert cosine term but this can be other importance sampling methods
        candidates[i].weight = rcpM * F_x * rcpUniformDistributionWeight; // Move 1.0 / M when computing weight. This is w_i (on page 9, 3.2)

        update(seed, reservoir, candidates[i], candidates[i].weight, F_x, i);
    }
}

vec3 RISReservoirSampling(vec3 pos, vec3 n, vec3 albedo)
{
    uint seed = uint(gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x) + gl_LaunchIDEXT.x;
    seed *= rtx.frameIndex;

    vec3 radiance = vec3(0.0);
    vec3 throughput = vec3(1.0);

    Reservoir reservoir;
    reservoir.totalWeights = 0.0;
    reservoir.W_y = 0.0;
    reservoir.index = -1;

    Candidate candidates[CANDIDATE_MAX];

    // Compute the weights of the candidates from the original distribution
    RISReservoir(reservoir, seed, pos, n, candidates, albedo);

    bool isValidIndex = reservoir.index > -1;

    if(isValidIndex) {
        // The selected light
        Candidate LightSource = reservoir.Y;
        uint pixelIndex = gl_LaunchIDEXT.y * 1280 + gl_LaunchIDEXT.x;

        //ReservoirStorageBuffer.reservoirs[pixelIndex] = r;

        // Compute the light weight to prevent bias
        // W_x = (sum(w_i) / M) / pdf(x)
        // Written as: 1 / pdf(x) * (1 / m * sum(w_i)), but remember 1 / pdf(x) and 1 / m is the same as dividing by them since 1 / x is rcp
        float rcpPDF = 1.0 / LightSource.Fx;

        // This is debug to ensure its valid, remove eventually
        if (isinf(rcpPDF))
            return vec3(1.0, 0.0, 1.0);

        // Evaluate the unbiased constribuion weight W_x
        // We moved rcpM = 1 / float(CANDIDATE_MAX) when computing weight for each candidate as suggested by paper
        float W_x = rcpPDF * (reservoir.totalWeights);
        reservoir.W_y = W_x;

        // Compute direct lighting using the elected light source
        float Visibility = CastShadowRay(pos, n, LightSource.lightDir, length(LightSource.light.LightPosition.xyz - pos) - 0.001);
        vec3 directLighting = computeDirectLighting(pos, n, albedo, LightSource.lightDir, LightSource.intensity, LightSource.light.LightColour.rgb) * W_x * Visibility;
        radiance += throughput * directLighting;
    }



    return radiance;
}

vec3 Verify(vec3 n, vec3 pos, vec3 albedo)
{
    vec3 throughput = vec3(1.0);
    uint pixelIndex = gl_LaunchIDEXT.y * 1280 + gl_LaunchIDEXT.x;
    vec3 pixelReservoir = texelFetch(InitialCandidatesImage, ivec2(gl_LaunchIDEXT.xy), 0).rgb;

    float W_y = pixelReservoir.y;

    int index = int(pixelReservoir.x);
    Light L = lightData.lights[index];

    vec3 LightDir = normalize(L.LightPosition.xyz - pos);
    float dist = length(L.LightPosition.xyz - pos);
    float att = 1.0 / (dist * dist);
    float intensity = 2000.0f * att;

    float Visibility = CastShadowRay(pos, n, LightDir, dist - 0.001);
    vec3 directLighting = computeDirectLighting(pos, n, albedo, LightDir, intensity, L.LightColour.rgb);
    vec3 radiance = throughput * directLighting * W_y * Visibility;

    return radiance;
}

float RadicalInverse_VdC(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 Hammersley(uint i, uint N)
{
    return vec2(float(i)/float(N), RadicalInverse_VdC(i));
}

vec2 DiskPoint(float sampleRadius, float x, float y)
{
	float r = sampleRadius * sqrt(x);
	float theta = y * (2.0 * PI);
	return vec2(r * cos(theta), r * sin(theta));
}


void update(uint seed, inout Reservoir reservoir, in float xi_weight, int index, int previousM)
{
    reservoir.totalWeights = reservoir.totalWeights + xi_weight;
    float r = GetRandomNumber(seed);
    reservoir.M += previousM;
    if(r < (xi_weight / reservoir.totalWeights))
    {
        reservoir.index = index;
    }
}

// @NOTE: Tip 3.4: Use 1 / M weights if and only if all inputs weights are identically distributed
// If initial candidates have different PDFs, such as when reusing across pixels. When reusing
// across pixels, if you're using different PDFs, the expectation is that nearby pixels might have used
// A different PDF compared to the others thus, MIS is needed to compute a balance heuristic.

Reservoir combine_reservoirs_spatial_reuse(vec4 current_pixel_reservoir_data, inout uint seed, vec3 n, vec3 pos)
{
    const int NUM_SPATIAL_NEIGHBOURS = 5; // 5 spatial neighbouring including the current pixel
    Reservoir reservoir;
    reservoir.index = -1;
    reservoir.W_y = 0.0;
    reservoir.M = 0;
    reservoir.totalWeights = 0.0;

    Reservoir neighbouring_reservoirs[NUM_SPATIAL_NEIGHBOURS]; // We will find 4 neighbours and fill this array

    // Init all reservoirs
    for(int i = 0; i < NUM_SPATIAL_NEIGHBOURS; i++)
    {
        neighbouring_reservoirs[i].index = -1;
        neighbouring_reservoirs[i].W_y = 0.0f;
        neighbouring_reservoirs[i].M = 0;
        neighbouring_reservoirs[i].totalWeights = 0.0f;
    }

    ivec2 current_pixel = ivec2(gl_LaunchIDEXT.xy);
    // Current pixel
    neighbouring_reservoirs[0].index = int(current_pixel_reservoir_data.x); // Current pixel index
    neighbouring_reservoirs[0].W_y = current_pixel_reservoir_data.y; // Current pixel weight
    neighbouring_reservoirs[0].M = int(current_pixel_reservoir_data.z); // Current pixel M

    for(uint i = 1; i < NUM_SPATIAL_NEIGHBOURS; i++)
    {
        vec2 rand = Hammersley(i, NUM_SPATIAL_NEIGHBOURS - 1);
        vec2 random = GetRandomHashValue01(seed);
        vec2 offset = DiskPoint(rtx.numPastFrames, random.x, random.y); // rtx.numPastFrame is controlling the radius of the disk, paper recommends 30 pixel radius

        ivec2 sample_pixel = current_pixel + ivec2(offset);

        ivec2 viewportSizeInt = ivec2(ubo.viewportSize);
        sample_pixel = clamp(sample_pixel, ivec2(0), viewportSizeInt - ivec2(1));

        neighbouring_reservoirs[i].index = int(texelFetch(TemporalReuseReservoirs, sample_pixel, 0).x);
        neighbouring_reservoirs[i].W_y   = texelFetch(TemporalReuseReservoirs, sample_pixel, 0).y;
        neighbouring_reservoirs[i].M     = min(int(texelFetch(TemporalReuseReservoirs, sample_pixel, 0).z), rtx.M); // 20 * int(current_pixel_reservoir_data.z)
    }

    // Update the reservoir using the neighbouring reservoirs
    for(uint i = 0; i < NUM_SPATIAL_NEIGHBOURS; i++)
    {
        Light L = lightData.lights[neighbouring_reservoirs[i].index];

        float dist = length(L.LightPosition.xyz - pos);
        const vec3 LightDir = normalize(L.LightPosition.xyz - pos);
        const float LightIntensity = 2000.0f * (1.0 / (dist * dist));

        // Evaluate F(x) at the current pixel
        float F_x = max(dot(n, LightDir), 0.0) * LightIntensity; // Simplied F(x) only doing diffuse

        // Algorithm 4: Line: 4: p^q(r.y) * r.W * r.M
        float w_i = F_x * neighbouring_reservoirs[i].W_y * neighbouring_reservoirs[i].M;

        // Update the reservoir using current sample data
        update(seed, reservoir, w_i, neighbouring_reservoirs[i].index, neighbouring_reservoirs[i].M);
    }

    return reservoir;
}


vec4 Spatial(vec3 n, vec3 pos, vec3 albedo)
{
    uint seed = uint(gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x) + gl_LaunchIDEXT.x;
    seed *= rtx.frameIndex;
    vec3 throughput = vec3(1.0);
    vec4 pixelReservoir = texelFetch(TemporalReuseReservoirs, ivec2(gl_LaunchIDEXT.xy), 0).rgba;

    Reservoir reservoir = combine_reservoirs_spatial_reuse(pixelReservoir, seed, n, pos);

    // The reservoir should now contain the new updated sample
    // Use the index from the reservoir to fetch the light data
    Light L = lightData.lights[reservoir.index];

    // Compute lighting using this light source
    vec3 LightDir = normalize(L.LightPosition.xyz - pos);
    float dist = length(L.LightPosition.xyz - pos);
    float att = 1.0 / (dist * dist);
    float intensity = 2000.0f * att;
    float Fx = max(dot(n, LightDir), 0.0) * intensity; // Simplied F(x) only doing diffuse

    reservoir.W_y = Fx > 0.0 ? (1.0 / Fx) * (1.0 / reservoir.M) * reservoir.totalWeights : 0.0;

    imageStore(SpatialReservoirStore, ivec2(gl_LaunchIDEXT.xy), vec4(reservoir.index, reservoir.W_y, reservoir.M, 0.0));

    float Visibility = CastShadowRay(pos, n, LightDir, dist - 0.001);
    vec3 directLighting = computeDirectLighting(pos, n, albedo, LightDir, intensity, L.LightColour.rgb);
    vec3 radiance = throughput * directLighting * reservoir.W_y * Visibility;

    return vec4(radiance, 0.0);
}


// @TODO: This pass is currently using the initial candidates and just sampling them directly to reproduce
// the same results as seen in the initial candidates pass to ensure the data in the texture is correct before we do spatial reuse
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

    //vec3 indirectLight = computeIndirectLightingBEFORENEW(worldPos, worldNormal, albedo, sunIntensity);
    // vec3 DirectLight = RISReservoirSampling(worldPos, worldNormal, albedo);
    //vec3 DirectLight = NaiveDirectLighting(worldPos, worldNormal, albedo);
    if(rtx.bounces == 1)
    {
        rayPayLoad.colour = vec4(Verify(worldNormal, worldPos, albedo), 1.0);
    } else {
        //vec3 radiance = vec3(Spatial(worldNormal, worldPos, albedo));
        rayPayLoad.colour = Spatial(worldNormal, worldPos, albedo);
    }
}