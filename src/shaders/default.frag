#version 450

layout(location = 0) in vec4 WorldPos;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec4 WorldNormal;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 brightColours;

layout(set = 0, binding = 0) uniform SceneUniform
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

struct Light
{
	int Type;
	vec4 LightPosition;
	vec4 LightColour;
	mat4 LightSpaceMatrix;
};

const int NUM_LIGHTS = 26;

layout(set = 0, binding = 1) uniform LightBuffer {
	Light lights[NUM_LIGHTS];
} lightData;

layout(push_constant) uniform Push
{
	mat4 ModelMatrix;
	vec4 BaseColourFactor;
	float Metallic;
	float Roughness;
}pc;

layout(set = 0, binding = 2) uniform sampler2DShadow shadowMap;

layout(set = 1, binding = 0) uniform sampler2D albedoTexture;
layout(set = 1, binding = 1) uniform sampler2D metallicRoughness;

#define PI 3.14159265359

// Fresnel (shlick approx)
vec3 Fresnel(vec3 halfVector, vec3 viewDir, vec3 baseColor, float metallic)
{
    vec3 F0 = vec3(0.04);
    F0 = (1 - metallic) * F0 + (metallic * baseColor);
    float HdotV = max(dot(halfVector, viewDir), 0.0);
    vec3 schlick_approx = F0 + (1 - F0) * pow(clamp(1 - HdotV, 0.0, 1.0), 5);
    return schlick_approx;
}

// Normal distribution function
float BeckmannNormalDistribution(vec3 normal, vec3 halfVector, float roughness)
{
    float a = roughness * roughness;
	float a2 = a * a; // alpha is roughness squared
	float NdotH = max(dot(normal, halfVector), 0.001); // preventing divide by zero
	float NdotHSquared = NdotH * NdotH;
	float numerator = exp((NdotHSquared - 1.0) / (a2 * NdotHSquared));
	float denominator = PI * a2 * (NdotHSquared * NdotHSquared); // pi * a2 * (n * h)^4

	float D = numerator / denominator;
	return D;
}

// Geometry term
float GeometryTerm(vec3 normal, vec3 halfVector, vec3 lightDir, vec3 viewDir)
{
	float NdotH = max(dot(normal, halfVector), 0.0);
	float NdotV = max(dot(normal, viewDir), 0.0);
	float VdotH = max(dot(viewDir, halfVector), 0.0);
	float NdotL = max(dot(normal, lightDir), 0.0);

	float term1 = 2 * (NdotH * NdotV) / VdotH;
	float term2 = 2 * (NdotH * NdotL) / VdotH;

	float G = min(1, min(term1, term2));

	return G;
}

// Compute BRDF
vec3 CookTorranceBRDF(vec3 normal, vec3 halfVector, vec3 viewDir, vec3 lightDir, float metallic, float roughness, vec3 baseColor, vec3 LightColour)
{
    vec3 F = Fresnel(halfVector, viewDir, baseColor, metallic);
    float D = BeckmannNormalDistribution(normal, halfVector, roughness);
	float G = GeometryTerm(normal, halfVector, lightDir, viewDir);

    vec3 L_Diffuse = (baseColor.xyz / PI) * (vec3(1,1,1) - F) * (1.0 - metallic);

    float NdotV = max(dot(normal, viewDir), 0.0);
	float NdotL = max(dot(normal, lightDir), 0.0);

	vec3 numerator = D * G * F;
	float denominator = (4 * NdotV * NdotL) + 0.001;

	vec3 specular = numerator / denominator;

    vec3 outLight = (L_Diffuse + specular) * LightColour.xyz * NdotL;

    return vec3(outLight);
}

// https://developer.nvidia.com/gpugems/gpugems/part-ii-lighting-and-shadows/chapter-11-shadow-map-antialiasing
float PCF(vec3 WorldPos)
{
	// Use direct lighting only. Point light shadows are handleded differently (cube depth)
	vec4 fragPositionInLightSpace = lightData.lights[0].LightSpaceMatrix * vec4(WorldPos, 1.0);
	fragPositionInLightSpace.xyz /= fragPositionInLightSpace.w;
	fragPositionInLightSpace.xy = fragPositionInLightSpace.xy * 0.5 + 0.5;

	vec2 texSize = 1.0 / textureSize(shadowMap, 0);
	int range = 2; // 4x4
	int samples = 0;
	float sum = 0.0;
	for(int x = -range; x < range; x++)
	{
		for(int y = -range; y < range; y++)
		{
			vec2 offset = vec2(x,y) * texSize;
			vec4 sampleCoord = vec4(fragPositionInLightSpace.xy + offset, fragPositionInLightSpace.z - 0.005, fragPositionInLightSpace.w);
			sum += textureProj(shadowMap, sampleCoord);
			samples++;
		}
	}

	return sum / float(samples);
}

void main()
{
	vec4 color = texture(albedoTexture, uv);
	vec3 emissive = vec3(0.0);
    vec3 wNormal = normalize(WorldNormal).xyz;

    // == Metal and Roughness ==
    float roughness = max(texture(metallicRoughness, uv).b, 0.1);
    float metallic = texture(metallicRoughness, uv).g;

    vec3 outLight = vec3(0.0);

	for(int i = 0; i < NUM_LIGHTS; i++)
	{
		vec3 lightDir = normalize(lightData.lights[i].LightPosition.xyz - WorldPos.xyz);
		vec3 viewDir = normalize(ubo.cameraPosition.xyz - WorldPos.xyz);
		vec3 halfVector = normalize(viewDir + lightDir);

		// is it a spot light?
		vec3 LightColour = vec3(0.0);
		bool isDirectional = lightData.lights[i].Type == 1 ? false : true;

		if(!isDirectional)
		{
			float dist = length(lightData.lights[i].LightPosition.xyz - WorldPos.xyz);

			float att = 1.0 / (dist * dist);
			LightColour = lightData.lights[i].LightColour.xyz * 5.0 * att;
		}
		else {
			LightColour = lightData.lights[i].LightColour.rgb * 20.0;
		}

		if(isDirectional) {
			float shadowTerm = 1.0 - PCF(WorldPos.xyz);
			outLight += shadowTerm * CookTorranceBRDF(wNormal, halfVector, viewDir, lightDir, metallic, roughness, color.xyz, LightColour);

		}
		else {
			outLight += CookTorranceBRDF(wNormal, halfVector, viewDir, lightDir, metallic, roughness, color.xyz, LightColour);
		}
	}

	float shadowTerm = 1.0 - PCF(WorldPos.xyz);
	vec3 ambient = vec3(0.1) * color.rgb;
	fragColor = vec4(vec3(wNormal), 1.0);

	float brightness = dot(fragColor.rgb, vec3(0.2126, 0.7152, 0.0722));
	if(brightness > 1.0)
		brightColours = vec4(clamp(fragColor.rgb, 0.0, 1.0), 1.0);
	else
		brightColours = vec4(0.0, 0.0, 0.0, 1.0);
}
