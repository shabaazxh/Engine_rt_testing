#pragma once

#include "Buffer.hpp"
#include <vector>
#include <volk/volk.h>
#include <array>
#include <glm/glm.hpp>
#include <utility>
#include "Utils.hpp"
#include "Image.hpp"

// TODO:
// Scene should release the resources of the GLTF
// GLTF Model class needs a move constructor and the GLTF mesh should be handled that way

// Moved Vertex into here since it's only relevant for for the GLTF Mesh to use 
namespace vk
{
	struct Vertex
	{
		glm::vec3 pos;
		glm::vec2 tex;
		glm::vec3 normal;
		std::array<uint8_t, 3> quaternion;

		static VkVertexInputBindingDescription GetBindingDescription()
		{
			VkVertexInputBindingDescription bindingDescrip{};
			bindingDescrip.binding = 0;
			bindingDescrip.stride = sizeof(Vertex);
			bindingDescrip.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

			return bindingDescrip;
		}

		static std::array<VkVertexInputAttributeDescription, 4> GetAttributeDescriptions()
		{
			std::array<VkVertexInputAttributeDescription, 4> attributes = {};

			attributes[0].binding = 0;
			attributes[0].location = 0;
			attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
			attributes[0].offset = offsetof(Vertex, pos);

			attributes[1].binding = 0;
			attributes[1].location = 1;
			attributes[1].format = VK_FORMAT_R32G32_SFLOAT;
			attributes[1].offset = offsetof(Vertex, tex);

			attributes[2].binding = 0;
			attributes[2].location = 2;
			attributes[2].format = VK_FORMAT_R32G32B32_SFLOAT;
			attributes[2].offset = offsetof(Vertex, normal);

			attributes[3].binding = 0;
			attributes[3].location = 3;
			attributes[3].format = VK_FORMAT_R8G8B8_UINT;
			attributes[3].offset = offsetof(Vertex, quaternion);

			return attributes;
		}
	};

	struct MeshPrimitive {
		std::vector<float> positions;
		std::vector<float> normals;
		std::vector<float> texcoords;
		// TODO: Bone weights
		std::vector<std::uint32_t> indices;
	};

	// Material, Material Manager are defined here for simplicity when testing and debugging 
	// they should be moved into their respective classes 

	class Context;
	class Material
	{
	public:
		// Take in textures array and load the Images inside material so material will be responsible 
		// for destroying those vulkan resources as well once finished 
		Material(Context& context);

		Material(Material&& other) noexcept : 
			context(other.context),
			textures(std::exchange(other.textures, {})),
			isValid(other.isValid)
		{}

		Material& operator=(Material&& other) noexcept {
			if (this != &other) {
				Destroy();  
				textures = std::move(other.textures);  
			}
			return *this;
		}


		void Destroy();

		std::vector<Image> textures; // make private in the future?
		bool isValid;
		float roughness;
		float metallic;
		glm::vec3 baseColourFactor;
	private:
		Context& context;
	};
//
//	struct MaterialHasher
//	{
//		size_t operator()(const Material& mat) const
//		{
//			return std::hash<std::string>()(mat.textures[0].name) ^ (std::hash<std::string>()(mat.textures[1].name) << 1);
//;		}
//	};

	struct MeshData;
	struct MaterialManager
	{
		// Material has the GPU textures 
		// We just need a descriptor for each material
		// Each mesh has a unique material index and it can index into descriptor array
		// to get and bind the correct material descriptor which will have its textures 
		std::vector<Material> materials;
		std::vector<VkDescriptorSet> materialDescriptorSets;
		// VkDescriptorSetLayout materialDescriptorSetLayout; // TODO: temp (needs to be discussed where this should go) 

		void Setup(Context& context); // creates the descriptor set layout by calling the method 
		void Destroy(Context& context); 
		void CreateDescriptorLayout(Context& context); // descriptor set layout for materials 
		void BuildMaterials(Context& context); // creates the materials

		uint32_t GetNextAvailableIndex() { 

			for (size_t i = 0; i < materials.size(); i++)
			{
				if (!materials[i].isValid)
				{
					return static_cast<uint32_t>(i);
				}
			}

			return -1;
		}

		void LoadTexturesForMaterial(uint32_t matIndex, const MeshData& mesh, Context& context);

		std::unordered_map<std::size_t, int> materialLookup; // Maps unique material hashes to material indices
	};

	// This is behaving like "Mesh" but I made my own so I did not edit
	// the pre-existing mesh structure 
	// This can be re-named to Mesh or moved into Mesh 
	// Needs to be more nicely organized 
	struct MeshData
	{
		const Context& context;
		std::vector<Vertex> vertices;
		std::vector<uint32_t> indices;
		Buffer vertexBuffer;
		Buffer indexBuffer;
		uint32_t materialIndex;
		std::vector<std::string> textures;

		MeshData(const Context& context);
		~MeshData();

		MeshData(const MeshData&) = delete;
		MeshData& operator=(const MeshData&) = delete;


		MeshData(MeshData&& other) noexcept
			:
			context(other.context),
			vertices(std::exchange(other.vertices, {})),  // Moves and clears the source
			indices(std::exchange(other.indices, {})),
			vertexBuffer(std::move(other.vertexBuffer)),  // Assuming Buffer supports move
			indexBuffer(std::move(other.indexBuffer)),
			materialIndex(std::move(other.materialIndex)),
			textures(std::move(other.textures))
		{
			//std::cout << "Move Constructing Image\n";

		}

		MeshData& operator=(MeshData&& other) noexcept {
			std::swap(vertices, other.vertices);
			std::swap(indices, other.indices);
			std::swap(vertexBuffer, other.vertexBuffer);
			std::swap(indexBuffer, other.indexBuffer);
			std::swap(materialIndex, other.materialIndex);
			std::swap(textures, other.textures);
			return *this;
		}
	};

	struct Mesh {
		std::string name;
		std::vector<MeshPrimitive> meshPrimitives;
		std::vector<Vertex> vertices;
		std::vector<uint32_t> indices;
	};

}


namespace vk
{
	// Defines a single GLTFModel for now 
	// Model consists of meshes 
	// Though we could continue to use it like this, a scene consists of many models
	// each model can be loaded and stored as this class type 
	class GLTFModel
	{
	public:
		GLTFModel(const Context& context) : context{ context } {}

		void Destroy() 
		{
			// This will invoke the destructor of all MeshData instances
			meshes.clear();
		}

		GLTFModel(const GLTFModel&) = delete;
		GLTFModel& operator=(const GLTFModel&) = delete;

		GLTFModel(GLTFModel&& other) noexcept :
			context(other.context),
			meshes(std::exchange(other.meshes, {})),
			name(other.name),
			position(other.position)
		{}

		GLTFModel& operator=(GLTFModel&& other) noexcept {
			std::swap(meshes, other.meshes);
			std::swap(name, other.name);
			std::swap(position, other.position);
			return *this;
		}

		const Context& context;
		std::vector<MeshData> meshes;
		glm::vec4 position = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
		std::string name;
	};

	
	GLTFModel LoadGLTF(const Context& context, const std::string& filepath);
	
}
