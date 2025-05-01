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
// 0.5, 0.75, 1.0 <- sky light float candidateWeight = candidatePdfG * intensity;
vec3 computeIndirectLighting(vec3 pos, vec3 n, vec3 albedo, float sunIntensity) {

    vec3 radiance = vec3(0.0);
    uint seed = uint(gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x) + gl_LaunchIDEXT.x;
    seed *= rtx.frameIndex;

    vec3 throughput = vec3(1.0);
    int bounces = nonuniformEXT(rtx.bounces);

    for (int bounce = 0; bounce < bounces; bounce++) {
        // ----- RIS for Direct Lighting -----
        const int candidateMax = 8;
        float totalWeights = 0.0f;
        Light selectedLight;
        float selectedWeight = 0.0f;
        float samplePdfG = 0.0f;
        bool isDirectional = false;
        float intensity = 0.0f;
        float lightWeight = 0.0f;

        // PDF of uniform distribution = 1 / total number of lights. Reciporal of that PDF is the light count e.g. 1 / 10 = 0.1 -> rcp = 1 / (1 / 10) = 10.0
        float candidateWeight = float(NUM_LIGHTS);

        for (int i = 0; i < candidateMax; i++) {

            int randomLightIndex = int(GetRandomNumber(seed) * float(NUM_LIGHTS));
            Light light = lightData.lights[randomLightIndex];

            vec3 lightDir;
            vec3 LightColour;
            isDirectional = (light.Type == 0) ? true : false; // 0 means directional

            if (isDirectional) {
                lightDir = normalize(light.LightPosition.xyz);
                LightColour = light.LightColour.rgb;
                intensity = sunIntensity;
            } else {
                lightDir = normalize(light.LightPosition.xyz - pos);
                float dist = length(light.LightPosition.xyz - pos);
                float att = 1.0 / (dist * dist);
                LightColour = light.LightColour.xyz * att;
                intensity = 1000.0f * att;
            }

            float candidatePdfG = intensity;
            const float candidateRISWeight = candidatePdfG * candidateWeight;
            totalWeights += candidateRISWeight;

            if (GetRandomNumber(seed) < (candidateRISWeight / totalWeights)) {
                selectedLight = light;
                selectedLight.LightColour.rgb = LightColour.rgb;
                selectedWeight = candidateRISWeight;
                samplePdfG = candidatePdfG;
            }
        }

        lightWeight = (totalWeights / float(candidateMax)) / samplePdfG;

        // Evaluate the selected light
        vec3 selectedDir = isDirectional == true ? normalize(selectedLight.LightPosition.xyz) : normalize(selectedLight.LightPosition.xyz - pos);
        float visibility = CastShadowRay(pos, n, selectedDir, isDirectional ? 10000.0 : length(selectedLight.LightPosition.xyz - pos) - 0.001);
        // I'm computing direct lighting here and then again once the light has been selected
        // Perhaps I should store this result and just use it after this loop ends so that directLighting
        // is the computed lighting contribution of the final selected light
        vec3 directLighting = computeDirectLighting(pos, n, albedo, selectedDir, intensity, selectedLight.LightColour.rgb) * lightWeight * visibility;
        radiance += throughput * directLighting;

        // ----- Indirect Lighting -----
        if (bounce == bounces - 1) break;

        vec3 omega_i = CosHemisphereDirection(n, pos, seed);

        diffusePayLoad.pos = vec3(0.0);
        diffusePayLoad.hitpos = vec3(0.0);
        diffusePayLoad.hit = -1.0;
        diffusePayLoad.colour = vec3(0.0);
        diffusePayLoad.hitnormal = vec3(0.0);
        traceRayEXT(topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT, 0xFF, 1, 0, 0, pos, 0.001, omega_i, 10000.0, 2);

        if (diffusePayLoad.hit > 0.0) {
            vec3 brdf = diffusePayLoad.colour / PI;
            float cosTheta = max(dot(diffusePayLoad.hitnormal, omega_i), 0.0);
            float pdf = cosTheta / PI;

            if (pdf > 0.001) {
                radiance += throughput *= (brdf * cosTheta) / pdf; // cos(theta) / pi PDF will cancel out with lambertian (need to change this)
            }

            pos = diffusePayLoad.hitpos;
            n = diffusePayLoad.hitnormal;
            albedo = diffusePayLoad.colour;

        } else {
            radiance += throughput * vec3(0.5, 0.75, 1.0); // Sky
            break;
        }
    }

    return radiance;
}

// RIS before new one
vec3 computeIndirectLightingBEFORENEW(vec3 pos, vec3 n, vec3 albedo, float sunIntensity) {

    vec3 radiance = vec3(0.0);
    uint seed = uint(gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x) + gl_LaunchIDEXT.x;
    seed *= rtx.frameIndex;

    vec3 throughput = vec3(1.0);

    int bounces = nonuniformEXT(rtx.bounces);

    for (int bounce = 0; bounce < bounces; bounce++) {
        // ----- Resampled Importance Sampling (RIS) -----

        // Define the maximum number of candidate lights for RIS
        const int candidateMax = 5;  // You can adjust this number for performance vs. quality

        // Accumulated total weight for lights
        float totalWeights = 0.0f;
        Light selectedLight;
        float samplePdfG = 0.0f;

        for (int i = 0; i < candidateMax; i++) {

            int randomLightIndex = int(GetRandomNumber(seed) * float(NUM_LIGHTS));
            Light light = lightData.lights[randomLightIndex];

            bool isDirectional = light.Type == 0 ? true : false;
            vec3 lightDir = vec3(0.0);
            vec3 LightColour = vec3(0.0);

            // Compute direction and visibility based on light type
            if (isDirectional) {
                lightDir = normalize(light.LightPosition.xyz);
                LightColour = light.LightColour.rgb;
            } else {
                lightDir = normalize(light.LightPosition.xyz - pos);
                float dist = length(light.LightPosition.xyz - pos);
                float att = 1.0 / (dist * dist);  // Attenuation factor for point lights
                LightColour = light.LightColour.xyz * att;
            }

            float intensity = isDirectional ? sunIntensity : 1000.0;  // Intensity based on light type
            // Compute the weight (importance) of this light source based on radiance
            float candidatePdfG = max(dot(n, lightDir), 0.0);  // Lambertian cosine term
            float candidateWeight = candidatePdfG * intensity;  // Weight based on radiance and cosine

            // Add the weighted contribution of this candidate light
            totalWeights += candidateWeight;

            // Perform weighted sampling (Reservoir Sampling)
            if (GetRandomNumber(seed) < (candidateWeight / totalWeights)) {
                selectedLight = light;
                selectedLight.LightColour.rgb = LightColour.rgb;
                samplePdfG = candidatePdfG;
            }
        }

        vec3 selectedDir = selectedLight.Type == 0 ? normalize(selectedLight.LightPosition.xyz) : normalize(selectedLight.LightPosition.xyz - pos);
        float visibility = CastShadowRay(pos, n, selectedDir, selectedLight.Type == 0 ? 10000.0 : length(selectedLight.LightPosition.xyz - pos) - 0.001);
        vec3 directLighting = computeDirectLighting(pos, n, albedo, selectedDir, selectedLight.Type == 0 ? sunIntensity : 1000.0, selectedLight.LightColour.rgb) * visibility;

        // After accumulating the weights, perform the final light selection
        radiance += throughput * directLighting * (totalWeights / candidateMax);

        // If we reach the last bounce, break early
        if (bounce == bounces - 1) {
            break;
        }

         // ----- Indirect Lighting: Compute new direction for bounce -----
        // Sample a new direction for indirect bounce based on the selected light
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

vec3 computeIndirectLightingOLD(vec3 pos, vec3 n, vec3 albedo, float sunIntensity) {

    vec3 radiance = vec3(0.0);
    uint seed = uint(gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x) + gl_LaunchIDEXT.x;
    seed *= rtx.frameIndex;

    vec3 throughput = vec3(1.0);

    int bounces = nonuniformEXT(rtx.bounces);

    for (int bounce = 0; bounce < bounces; bounce++)
    {
        for(int i = 0; i < NUM_LIGHTS; i++)
        {
            Light light = lightData.lights[i];

            bool isDirectional = light.Type == 1 ? false : true;
            vec3 lightDir = vec3(0.0);
            vec3 LightColour = vec3(0.0);
            float intensity = 0.0;
            float visibility = 1.0;
            // Check if the light is directional

            if(isDirectional)
            {
                lightDir = normalize(light.LightPosition.xyz);
                LightColour = light.LightColour.rgb;
                intensity = sunIntensity;
                visibility = CastShadowRay(pos, n, lightDir, 10000.0); // compute visibility only for directional light source
            } else
            {
                lightDir = normalize(light.LightPosition.xyz - pos);
                float dist = length(light.LightPosition.xyz - pos);
                float att = 1.0 / (dist * dist);
                LightColour = light.LightColour.xyz * att;
                intensity = 1000.0f;
                visibility = CastShadowRay(pos, n, lightDir, dist);
            }

            vec3 directLight = computeDirectLighting(
                pos,
                n,
                albedo,
                lightDir,
                intensity,
                LightColour
            );

            radiance += throughput * directLight * visibility;
        }

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


#define CANDIDATE_MAX 8

struct Candidate
{
    Light light;
    float Fx;
    float weight;
    vec3 lightDir;
    float intensity;
};


int RandomIndex(uint seed, in Candidate candidates[CANDIDATE_MAX], in float totalWeights)
{
    float r = GetRandomNumber(seed);
    float accum = 0.0f;

    for(int i = 0; i < CANDIDATE_MAX; i++)
    {
        if(candidates[i].weight > 0.0)
        {
            r = r - (candidates[i].weight / totalWeights);
            if(r <= 0.0)
            {
                return i;
            }
        }
    }

    return -1;
}

void ComputeCandidatesWeights(in uint seed, vec3 pos, vec3 n, inout Candidate candidates[CANDIDATE_MAX], inout float totalWeights)
{
    const float rcpUniformDistributionWeight = float(NUM_LIGHTS); // PDF of uniform distribution = 1 / total number of lights. Reciporal of that PDF is the light count e.g. 1 / 10 = 0.1 -> rcp = 1 / (1 / 10) = 10.0
    const float rcpM = 1.0 / float(CANDIDATE_MAX);

    // Picking any light direction has a uniform distribution
    for (int i = 0; i < CANDIDATE_MAX; i++) {

        // Pick a random light from all lights
        int randomLightIndex = int(GetRandomNumber(seed) * float(NUM_LIGHTS));
        Light light = lightData.lights[randomLightIndex];
        candidates[i].light = light;

        // Get the properties of this light
        candidates[i].lightDir = normalize(light.LightPosition.xyz - pos);
        float dist = length(light.LightPosition.xyz - pos);
        float att = 1.0 / (dist * dist);
        candidates[i].intensity = 1000.0f * att;

        // Compute RIS weight for this candidate light
        float F_x = max(dot(n, candidates[i].lightDir), 0.0) + 0.001; // Simplied F(x) for weighting. Not sure if need to compute entir BRDF * cosine * ....?
        candidates[i].Fx = F_x; // The target function F(x) that PDF(X) approximates better with more candidates. Using lambert cosine term but this can be other importance sampling methods
        float candidateRISWeight = rcpM * F_x * rcpUniformDistributionWeight;
        candidates[i].weight = candidateRISWeight; // Move 1.0 / M when computing weight as suggested
        totalWeights += candidateRISWeight;
    }
}

vec3 RISSampling(vec3 pos, vec3 n, vec3 albedo)
{
    uint seed = uint(gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x) + gl_LaunchIDEXT.x;
    seed *= rtx.frameIndex;

    vec3 radiance = vec3(0.0);
    vec3 throughput = vec3(1.0);
    float totalWeights = 0.0f;

    Candidate candidates[CANDIDATE_MAX];

    // Compute the weights of the candidates from the original distribution
    ComputeCandidatesWeights(seed, pos, n, candidates, totalWeights);

    // Resample from the new distribution and select a light source proportional to its weight w_i
    // This will give us a sample X drawn from a PDF that is approximately proportional to the target function F(x)
    int selectedIndex = RandomIndex(seed, candidates, totalWeights);

    // Ensure the index is valid before using it
    bool isValidIndex = selectedIndex != -1;

    // If the index is valid, we can use it to compute the radiance
    if(isValidIndex)
    {
        // The selected light
        Candidate LightSource = candidates[selectedIndex];
        // Compute the light weight to prevent bias
        // W_x = (sum(w_i) / M) / pdf(x)
        // Written as: 1 / pdf(x) * (1 / m * sum(w_i)), but remember 1 / pdf(x) and 1 / m is the same as dividing by them since 1 / x is rcp
        //float LightWeight = (totalWeights / float(CANDIDATE_MAX)) / LightSource.samplePdfG;
        float rcpPDF = 1.0 / LightSource.Fx;
        float rcpM = 1.0 / float(CANDIDATE_MAX); // Moved this out of the end, instead account for it during weight computation
        // Evaluate the unbiased constribuion weight W_x
        float W_x = rcpPDF * (totalWeights);

        // Compute direct lighting using the elected light source
        //vec3 DiffuseBRDF(vec3 normal, vec3 position, vec3 albedo, vec3 LightDirection, vec3 LightColour, float LightIntensity)

        float Visibility = CastShadowRay(pos, n, LightSource.lightDir, length(LightSource.light.LightPosition.xyz - pos) - 0.001);
        vec3 directLighting = DiffuseBRDF(n, pos, albedo, LightSource.lightDir.xyz, LightSource.light.LightColour.rgb, LightSource.intensity);
        //vec3 directLighting = computeDirectLighting(pos, n, albedo, LightSource.lightDir, LightSource.intensity, LightSource.light.LightColour.rgb) * W_x * Visibility;
        radiance += throughput * directLighting;
    }


    return radiance;
}

struct Reservoir
{
    Candidate Y;
    float W_y;
    float totalWeights;
    int index;
};

void update(uint seed, inout Reservoir reservoir, in Candidate xi, in float xi_weight, int index)
{
    reservoir.totalWeights = reservoir.totalWeights + xi_weight;
    float r = GetRandomNumber(seed);
    if(r < (xi_weight / reservoir.totalWeights))
    {
        reservoir.Y = xi;
        reservoir.index = index;
    }
}

void RISReservoir(inout Reservoir reservoir, uint seed, vec3 pos, vec3 n, inout Candidate candidates[CANDIDATE_MAX], vec3 albedo)
{
    const float rcpUniformDistributionWeight = float(NUM_LIGHTS); // PDF of uniform distribution = 1 / total number of lights. Reciporal of that PDF is the light count e.g. 1 / 10 = 0.1 -> rcp = 1 / (1 / 10) = 10.0
    const float rcpM = 1.0 / float(CANDIDATE_MAX);

    // Picking any light direction has a uniform distribution
    for (int i = 0; i < CANDIDATE_MAX; i++) {

        // Pick a random light from all lights
        int randomLightIndex = int(GetRandomNumber(seed) * float(NUM_LIGHTS));
        candidates[i].light = lightData.lights[randomLightIndex];

        // Get the properties of this light
        float dist = length(candidates[i].light.LightPosition.xyz - pos);
        float att = 1.0 / (dist * dist);
        candidates[i].lightDir = normalize(candidates[i].light.LightPosition.xyz - pos);
        candidates[i].intensity = 1000.0f * att;

        // Compute RIS weight for this candidate light
        float F_x = max(dot(n, candidates[i].lightDir), 0.001) * candidates[i].intensity; // Simplied F(x) for weighting. Not sure if need to compute entir BRDF * cosine * ....?
        candidates[i].Fx = F_x; // The target function F(x) that PDF(X) approximates better with more candidates. Using lambert cosine term but this can be other importance sampling methods
        candidates[i].weight = rcpM * F_x * rcpUniformDistributionWeight; // Move 1.0 / M when computing weight as suggested

        update(seed, reservoir, candidates[i], candidates[i].weight, i);
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

        // Store the reservoirs index into the correct light + the weighting of it
        // Then in the next pass, e.g. compute pass, spatially reuse informaion to compute a new reservoir
        // if the next pass was temporal reuse, then we would take the previous temporal + spatial output and mix it with current intial candidates
        // once this is done, we pass it to another pass which will then spatially reuse and from there pass it to the next RT pass for it to be used
        // and select reservoir for current pixel
        imageStore(ReservoirsImage, ivec2(gl_LaunchIDEXT.xy), vec4(reservoir.index, reservoir.W_y, 0.0, 0.0));


        // Compute direct lighting using the elected light source
        float Visibility = CastShadowRay(pos, n, LightSource.lightDir, length(LightSource.light.LightPosition.xyz - pos) - 0.001);
        vec3 directLighting = computeDirectLighting(pos, n, albedo, LightSource.lightDir, LightSource.intensity, LightSource.light.LightColour.rgb) * W_x * Visibility;
        radiance += throughput * directLighting;
    }

    return radiance;
}



//vec3 RISDirectLighting(vec3 pos, vec3 n, vec3 albedo)
//{
//    uint seed = uint(gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x) + gl_LaunchIDEXT.x;
//    seed *= rtx.frameIndex;
//
//    vec3 radiance = vec3(0.0);
//    vec3 throughput = vec3(1.0);
//
//    //Candidiate candidates[candidateMax];
//
//    float totalWeights = 0.0f;
//    Light selectedLight;
//    float selectedWeight = 0.0f;
//    float samplePdfG = 0.0f;
//    bool isDirectional = false;
//    float intensity = 0.0f;
//    float lightWeight = 0.0f;
//    vec3 selectedDir = vec3(0);
//
//    // Picking any light direction has a uniform distribution
//    // Each light is equally likely to be picked as any other. PDF = 1 / NUM_LIGHTS
//    // PDF of uniform distribution = 1 / total number of lights. Reciporal of that PDF is the light count e.g. 1 / 10 = 0.1 -> rcp = 1 / (1 / 10) = 10.0
//    float candidateWeight = float(NUM_LIGHTS);
//
//    // For N number of candidate lights
//    for (int i = 0; i < CANDIDATE_MAX; i++) {
//
//        // Pick a random light from all lights
//        int randomLightIndex = int(GetRandomNumber(seed) * float(NUM_LIGHTS));
//        Light light = lightData.lights[randomLightIndex];
//        //candidates[i].light = light;
//
//        // Get the properties of this light
//        vec3 lightDir = normalize(light.LightPosition.xyz - pos);
//        float dist = length(light.LightPosition.xyz - pos);
//        float att = 1.0 / (dist * dist);
//        vec3 LightColour = light.LightColour.xyz;
//        intensity = 1000.0f * att;
//
//        // Compute RIS weight for this candidate light
//        float candidatePdfG = max(dot(n, lightDir), 0.0); // Weight based on lamberts cosine to see contribution of this light at this position
//        const float candidateRISWeight = candidatePdfG * candidateWeight;
//        totalWeights += candidateRISWeight;
//
//        // To prevent bias, randomly select a light based on its weight
//        if (GetRandomNumber(seed) < (candidateRISWeight / totalWeights)) {
//            selectedLight = light;
//            selectedLight.LightColour.rgb = LightColour.rgb;
//            selectedWeight = candidateRISWeight;
//            samplePdfG = candidatePdfG;
//            selectedDir = lightDir;
//        }
//    }
//
//    lightWeight = (totalWeights / float(CANDIDATE_MAX)) / samplePdfG;
//
//    // Compute direct lighting usinng the selected light [Check lightWeight here. Is it correct?]
//    float visibility = CastShadowRay(pos, n, selectedDir, length(selectedLight.LightPosition.xyz - pos) - 0.001);
//    vec3 directLighting = computeDirectLighting(pos, n, albedo, selectedDir, intensity, selectedLight.LightColour.rgb) * lightWeight * visibility;
//    radiance += throughput * directLighting;
//
//    return radiance;
//}


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
        intensity = 1000.0f * att;

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