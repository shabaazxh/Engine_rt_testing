#version 450

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

layout(push_constant) uniform Push
{
	mat4 ModelMatrix;
	vec4 BaseColourFactor;
	float Metallic;
	float Roughness;
}pc;

layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 tex;
layout(location = 2) in vec3 normal;
layout(location = 3) in uvec3 compressedTBN;

layout(location = 0) out vec4 WorldPos;
layout(location = 1) out vec2 uv;
layout(location = 2) out vec4 WorldNormal;
layout(location = 3) out mat3 TBN;

float unpack8bitToFloat(uint value)
{
	float normalized = float(value) / 255.0;
	float f = (normalized * 2.0f) - 1.0f;
	return f;
}

vec4 unpackQuaternion(uvec3 packed) {
    
    vec3 quat = vec3(
        unpack8bitToFloat(packed.x),
        unpack8bitToFloat(packed.y),
        unpack8bitToFloat(packed.z)
    );
    float w = sqrt(1.0 - dot(quat, quat));
    return vec4(quat, w);
}

// Reference: 
// Can reduce instructions by not reconstructing normal
// We already the world normal available to us in the shader
mat3 QuatToMat3(vec4 q) {
    
    float x = q.x;
    float y = q.y;
    float z = q.z;
    float w = q.w;

    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;
    float xy = x * y;
    float xz = x * z;
    float yz = y * z;
    float wx = w * x;
    float wy = w * y;
    float wz = w * z;

    mat3 result;
    result[0][0] = 1.0 - 2.0 * (y2 + z2);
    result[0][1] = 2.0 * (xy - wz);
    result[0][2] = 2.0 * (xz + wy);

    result[1][0] = 2.0f * (xy + wz);
    result[1][1] = 1.0 - 2.0f * (x2 + z2);
    result[1][2] = 2.0 * (yz + wx);

    return result;
}

void main()
{
	vec4 quaternion = normalize(unpackQuaternion(compressedTBN));
	mat3 tbnMatrix = QuatToMat3(quaternion);

	WorldNormal = normalize(pc.ModelMatrix * vec4(normal, 0.0));

    vec3 T = normalize((pc.ModelMatrix * vec4(tbnMatrix[0], 0.0)).xyz);
    vec3 B = normalize((pc.ModelMatrix * vec4(tbnMatrix[1], 0.0)).xyz);
    
    // Reference: Tangent Frame Transformation with Dual-Quaternion (Slide 23)
    B *= sign(quaternion.w); // handededness 

    TBN = mat3(T, B, WorldNormal); 

	uv = tex;
	WorldPos = pc.ModelMatrix * vec4(pos, 1.0);
	gl_Position = ubo.projection * ubo.view * pc.ModelMatrix * vec4(pos, 1.0);
}
