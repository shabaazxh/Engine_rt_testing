#include "Scene.hpp"
#include <unordered_map>

vk::Scene::Scene(Context& context, MaterialManager& materialManager) : context(context), materialManager{ materialManager }
{
	m_LightUBO.resize(MAX_FRAMES_IN_FLIGHT);
	// Light uniform buffers
	for (auto& buffer : m_LightUBO)
		buffer = CreateBuffer("LightUBO", context, sizeof(LightBuffer), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

}

void vk::Scene::AddModel(GLTFModel& GLTF, MaterialManager& materialManager)
{
	// Check if the material index this mesh refers to is already in-use
	for (auto& mesh : GLTF.meshes)
	{
		if (materialManager.materialLookup[mesh.materialIndex] > 0)
		{
			// This index is already in use
			// How can i check if this is a unique material for the mesh or an existing one ?
			auto& material = materialManager.materials[mesh.materialIndex];
			bool isSameAlbedo   = material.textures[0].name == mesh.textures[0];
			bool isSameMetRough = material.textures[1].name == mesh.textures[1];

			if (isSameAlbedo && isSameMetRough)
			{
				materialManager.materialLookup[mesh.materialIndex]++; // just keeping track of how many meshes might be using thi could be useful someday
				continue; // material is the same, just re-use it
			}
			else
			{
				uint32_t newIndex = materialManager.GetNextAvailableIndex();
				assert(newIndex != -1);
;				materialManager.materials[newIndex].textures.resize(mesh.textures.size());
				for (size_t i = 0; i < mesh.textures.size(); i++) {
					VkFormat FORMAT = i == 0 ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM; // index 0 is albedo, the rest should use UNORM
					materialManager.materials[newIndex].textures[i] = (std::move(LoadTextureFromDisk(mesh.textures[i], context, FORMAT)));
				}

				materialManager.materials[newIndex].isValid = true;
				materialManager.materialLookup[newIndex] = 1;
				mesh.materialIndex = newIndex;
			}
		}
		else
		{
			materialManager.materials[mesh.materialIndex].textures.resize(mesh.textures.size());

			for (size_t i = 0; i < mesh.textures.size(); i++) {
				VkFormat FORMAT = i == 0 ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM; // index 0 is albedo, the rest should use UNORM
				materialManager.materials[mesh.materialIndex].textures[i] = (std::move(LoadTextureFromDisk(mesh.textures[i], context, FORMAT)));
			}
			materialManager.materials[mesh.materialIndex].isValid = true;
			materialManager.materialLookup[mesh.materialIndex] = 1;
		}
	}


	// collect all of the texture paths
	// we need to create a material and when we push
	// theres a material per-mesh, we need to be able to index into a material, get it's albedo index to index into an array of textures
	// TODO: @IMPROVE -> check if a pre-existing albedo texture with same name exists, if so just use it's index for current materials albedo
	std::vector<std::string> texture_paths;
	materialsRT.resize(GLTF.meshes.size()); // each mesh has a unique material for now
	for (size_t meshIndex = 0; meshIndex < GLTF.meshes.size(); meshIndex++)
	{
		MaterialRT materialRT = {};
		const auto& meshData = GLTF.meshes[meshIndex];
		for (size_t i = 0; i < 1; i++) // should be looping until meshData.textures.size() but we're collecting only albedo for now
		{
			texture_paths.push_back(meshData.textures[i]);
			materialRT.albedoIndex = static_cast<uint32_t>(texture_paths.size() - 1);
		}

		// if mesh 0 is drawn first, it'll index into material[0] and get the indexes pointing to the right textures in the texture array
		materialsRT[meshIndex] = std::move(materialRT);
		//meshData.materialIndex = static_cast<uint32_t>(mesh); // material index should be the same as the order the meshes appear

	}

	// now begin to load the textures
	textures.resize(texture_paths.size());
	// TODO: @PROBLEM: All textures are being loaded as SRGB, which is wrong for roughness and metallic. Ok for now since we're loading only albedo
	for (size_t i = 0; i < texture_paths.size(); i++)
	{
		textures[i] = std::move(LoadTextureFromDisk(texture_paths[i], context, VK_FORMAT_R8G8B8A8_SRGB));
	}

	// now i need a descriptor pool which allows updating after binding which allows indexing
	// use VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT_EXT as flag?

	std::vector<Vertex> allVerts;
	std::vector<uint32_t> allIndices;
	std::vector<glm::uvec2> meshOffsets;
	std::vector<uint32_t> meshIDs;

	uint32_t vertexOffset = 0;
	uint32_t indexOffset = 0;
	uint32_t currentMeshID = 0;
	for (auto& mesh : GLTF.meshes)
	{
		VkDeviceSize vertexSize = sizeof(mesh.vertices[0]) * mesh.vertices.size();
		CreateAndUploadBuffer(context, mesh.vertices.data(), vertexSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh.vertexBuffer);

		VkDeviceSize indexSize = sizeof(mesh.indices[0]) * mesh.indices.size();
		CreateAndUploadBuffer(context, mesh.indices.data(), indexSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, mesh.indexBuffer);

		meshOffsets.push_back(glm::uvec2(indexOffset, vertexOffset));

		allVerts.insert(allVerts.end(), mesh.vertices.begin(), mesh.vertices.end());
		allIndices.insert(allIndices.end(), mesh.indices.begin(), mesh.indices.end());

		vertexOffset += mesh.vertices.size();
		indexOffset  += mesh.indices.size();
	}


	// now create the large vertex, index + mesh offset buffers
	VkDeviceSize vertSize = sizeof(allVerts[0]) * allVerts.size();
	CreateAndUploadBuffer(context, allVerts.data(), vertSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vertexBuffer);

	VkDeviceSize indexSize = sizeof(allIndices[0]) * allIndices.size();
	CreateAndUploadBuffer(context, allIndices.data(), indexSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, indexBuffer);

	VkDeviceSize offsetSize = sizeof(meshOffsets[0]) * meshOffsets.size();
	CreateAndUploadBuffer(context, meshOffsets.data(), offsetSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, meshOffsetBuffer);

	/*
	*
	*  This will give us an array of materials e.g.
	* struct Material
	* {
	*	uint albedoIndex;
	* }
	* layout(binding = x) materials[]; <- we can then index into this using the mesh instance
	*/
	VkDeviceSize materialSize = sizeof(materialsRT[0]) * materialsRT.size();
	CreateAndUploadBuffer(context, materialsRT.data(), materialSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, RTMaterialsBuffer);

	gltfModels.push_back(std::move(GLTF));

	CreateBLAS();
	CreateTLAS();
}

// Maybe do it for a single mesh for now ?
void vk::Scene::CreateBLAS()
{
	const uint32_t numMeshes = gltfModels[0].meshes.size();
	//std::vector<VkAccelerationStructureGeometryKHR> geometries;
	//geometries.reserve(numMeshes);
	//std::vector<uint32_t> primitiveCount;
	//primitiveCount.reserve(numMeshes);

	std::vector<VkAccelerationStructureBuildRangeInfoKHR> buildInfos;
	buildInfos.reserve(numMeshes);
	BottomLevelAccelerationStructures.resize(numMeshes);

	// First pass: collect all geometries and primitive counts
	for (size_t i = 0; i < numMeshes; i++) {

		auto& mesh = gltfModels[0].meshes[i];
		VkDeviceAddress vertexBufferAddress = GetBufferDeviceAddress(context.device, mesh.vertexBuffer.buffer);
		VkDeviceAddress indexBufferAddress = GetBufferDeviceAddress(context.device, mesh.indexBuffer.buffer);

		const uint32_t numPrims = static_cast<uint32_t>(mesh.indices.size() / 3);
		//primitiveCount.push_back(numPrims);

		VkAccelerationStructureGeometryTrianglesDataKHR triangles{
			.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
			.vertexFormat = VK_FORMAT_R32G32B32A32_SFLOAT,
			.vertexData = {.deviceAddress = vertexBufferAddress},
			.vertexStride = sizeof(Vertex),
			.maxVertex = static_cast<uint32_t>(mesh.vertices.size() - 1),
			.indexType = VK_INDEX_TYPE_UINT32,
			.indexData = {.deviceAddress = indexBufferAddress},
			.transformData = {.deviceAddress = 0}
		};

		VkAccelerationStructureGeometryKHR geometry{
			.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
			.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
			.geometry = {.triangles = triangles},
			.flags = VK_GEOMETRY_OPAQUE_BIT_KHR
		};

		VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo =
		{
			.primitiveCount = numPrims, // num tri
			.primitiveOffset = 0,
			.firstVertex = 0,
			.transformOffset = 0
		};

		// Build geometry info for all meshes at once
		VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo{
			.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
			.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
			.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
			.geometryCount = static_cast<uint32_t>(1),
			.pGeometries = &geometry
		};

		// Get size requirements for the entire BLAS
		VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo{
			.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR
		};

		vkGetAccelerationStructureBuildSizesKHR(
			context.device,
			VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
			&accelerationStructureBuildGeometryInfo,
			&numPrims,
			&accelerationStructureBuildSizesInfo
		);

		printf("Acceleration structure size: %zu\n", size_t(accelerationStructureBuildSizesInfo.accelerationStructureSize));
		printf("Build Scratch size: %zu\n", size_t(accelerationStructureBuildSizesInfo.buildScratchSize));

		BottomLevelAccelerationStructures[i].buffer = std::make_unique<Buffer>(
			CreateBuffer(
				"BLASBuffer",
				context,
				accelerationStructureBuildSizesInfo.accelerationStructureSize,
				VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
				VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT)
		);

		// Create acceleration structure
		VkAccelerationStructureCreateInfoKHR accelerationStructureCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
			.buffer = BottomLevelAccelerationStructures[i].buffer->buffer,
			.size = accelerationStructureBuildSizesInfo.accelerationStructureSize,
			.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR
		};
		vkCreateAccelerationStructureKHR(context.device, &accelerationStructureCreateInfo, nullptr, &BottomLevelAccelerationStructures[i].handle);


		Buffer scratchBuffer = CreateBuffer(
			"ScratchBuffer",
			context,
			accelerationStructureBuildSizesInfo.buildScratchSize,
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
			VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
		);

		VkAccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo = {
			.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
			.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
			.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
			.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
			.dstAccelerationStructure = BottomLevelAccelerationStructures[i].handle,
			.geometryCount = static_cast<uint32_t>(1),
			.pGeometries = &geometry,
			.scratchData = {.deviceAddress = GetBufferDeviceAddress(context.device, scratchBuffer.buffer)}
		};


		std::vector<VkAccelerationStructureBuildRangeInfoKHR*> accelerationBuildStructureInfos = {
			&accelerationStructureBuildRangeInfo
		};

		ExecuteSingleTimeCommands(context, [&](VkCommandBuffer cmd)
			{
				vkCmdBuildAccelerationStructuresKHR(
					cmd,
					1,
					&accelerationBuildGeometryInfo,
					accelerationBuildStructureInfos.data());
			}
		);

		scratchBuffer.Destroy(context.device);
		//geometries.push_back(geometry);
		//buildInfos.push_back(accelerationStructureBuildRangeInfo);

		VkAccelerationStructureDeviceAddressInfoKHR accelerationDeviceAddressInfo = {
			.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
			.accelerationStructure = BottomLevelAccelerationStructures[i].handle
		};


		BottomLevelAccelerationStructures[i].deviceAddress = vkGetAccelerationStructureDeviceAddressKHR(
			context.device,
			&accelerationDeviceAddressInfo
		);
	}
}

void vk::Scene::CreateTLAS()
{
	VkTransformMatrixKHR transform_matrix = {
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f
	};

	std::vector<VkAccelerationStructureInstanceKHR> instances;

	for (size_t i = 0; i < BottomLevelAccelerationStructures.size(); i++)
	{
		VkAccelerationStructureInstanceKHR accelerationStructureInstance = {
			.transform = transform_matrix,
			.instanceCustomIndex = static_cast<uint32_t>(i),
			.mask = 0xFF,
			.instanceShaderBindingTableRecordOffset = 0,
			.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
			.accelerationStructureReference = BottomLevelAccelerationStructures[i].deviceAddress
		};

		instances.push_back(accelerationStructureInstance);
	}


	instanceBuffer = std::make_unique<Buffer>(
		CreateBuffer(
			"InstanceBuffer",
			context,
			sizeof(VkAccelerationStructureInstanceKHR) * instances.size(),
			VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		)
	);

	instanceBuffer->WriteToBuffer(instances.data(), sizeof(VkAccelerationStructureInstanceKHR) * instances.size());

	VkDeviceOrHostAddressConstKHR instanceDataDeviceAddress =
	{
		.deviceAddress = GetBufferDeviceAddress(context.device, instanceBuffer->buffer)
	};

	VkAccelerationStructureGeometryKHR accelerationStructureGeometry = {};
	accelerationStructureGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
	accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
	accelerationStructureGeometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
	accelerationStructureGeometry.geometry.instances.arrayOfPointers = VK_FALSE;
	accelerationStructureGeometry.geometry.instances.data = instanceDataDeviceAddress;

	VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo = {};
	accelerationStructureBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	accelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	accelerationStructureBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	accelerationStructureBuildGeometryInfo.geometryCount = 1;
	accelerationStructureBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;

	const uint32_t primitiveCount = static_cast<uint32_t>(instances.size());

	VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR
	};

	vkGetAccelerationStructureBuildSizesKHR(
		context.device,
		VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		&accelerationStructureBuildGeometryInfo,
		&primitiveCount,
		&accelerationStructureBuildSizesInfo
	);

	TopLevelAccelerationStructure.buffer = std::make_unique<Buffer>(
		CreateBuffer(
			"TLASBuffer",
			context,
			accelerationStructureBuildSizesInfo.accelerationStructureSize,
			VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
			VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT)
	);

	// Create the top-level AS
	VkAccelerationStructureCreateInfoKHR accelerationStructureCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
		.buffer = TopLevelAccelerationStructure.buffer->buffer,
		.size = accelerationStructureBuildSizesInfo.accelerationStructureSize,
		.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
	};
	vkCreateAccelerationStructureKHR(context.device, &accelerationStructureCreateInfo, nullptr, &TopLevelAccelerationStructure.handle);

	// Binding
	Buffer scratchBuffer = CreateBuffer(
		"ScratchBuffer",
		context,
		accelerationStructureBuildSizesInfo.buildScratchSize,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
	);

	VkAccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
		.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
		.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
		.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
		.dstAccelerationStructure = TopLevelAccelerationStructure.handle,
		.geometryCount = 1,
		.pGeometries = &accelerationStructureGeometry,
		.scratchData = {.deviceAddress = GetBufferDeviceAddress(context.device, scratchBuffer.buffer)}
	};


	VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo =
	{
		.primitiveCount = static_cast<uint32_t>(instances.size()), // num inst
		.primitiveOffset = 0,
		.firstVertex = 0,
		.transformOffset = 0
	};

	std::vector< VkAccelerationStructureBuildRangeInfoKHR*> accelerationBuildStructureInfos = {
		&accelerationStructureBuildRangeInfo
	};

	ExecuteSingleTimeCommands(context, [&](VkCommandBuffer cmd)
		{
			vkCmdBuildAccelerationStructuresKHR(
				cmd,
				1,
				&accelerationBuildGeometryInfo,
				accelerationBuildStructureInfos.data());
		}
	);

	scratchBuffer.Destroy(context.device);
	//instanceBuffer->Destroy(context.device);

	VkAccelerationStructureDeviceAddressInfoKHR accelerationDeviceAddressInfo = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
		.accelerationStructure = TopLevelAccelerationStructure.handle
	};

	TopLevelAccelerationStructure.deviceAddress = vkGetAccelerationStructureDeviceAddressKHR(
		context.device,
		&accelerationDeviceAddressInfo
	);
}

// This should really be called RenderMeshes which renders meshes in the scene
void vk::Scene::DrawGLTF(VkCommandBuffer cmd, VkPipelineLayout pipelineLayout)
{
	for (auto& model : gltfModels)
	{
		for (auto& mesh : model.meshes)
		{
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 1, 1, &materialManager.materialDescriptorSets[mesh.materialIndex], 0, nullptr);

			MeshPushConstants pc = {};
			pc.ModelMatrix = glm::mat4(1.0f);

			// Doing this for now. Will update in the future
			if (model.name.find("Sponza") != std::string::npos)
			{
				pc.ModelMatrix = glm::scale(pc.ModelMatrix, glm::vec3(0.005, 0.005, 0.005));
			}
			else
			{
				pc.ModelMatrix = glm::translate(pc.ModelMatrix, glm::vec3(model.position)); // Apply translation first
				pc.ModelMatrix = glm::scale(pc.ModelMatrix, glm::vec3(0.3f, 0.3f, 0.3f)); // Then apply scaling
			}


			vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(MeshPushConstants), &pc);
			// Set up push constants
			VkDeviceSize offset[] = { 0 };
			vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertexBuffer.buffer, offset);
			vkCmdBindIndexBuffer(cmd, mesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdDrawIndexed(cmd, static_cast<uint32_t>(mesh.indices.size()), 1, 0, 0, 0);
		}
	}
}


// TODO: Sort and implement these
void vk::Scene::RenderFrontMeshes(VkCommandBuffer cmd, VkPipelineLayout pipelineLayout)
{

}
void vk::Scene::RenderBackMeshes(VkCommandBuffer cmd, VkPipelineLayout pipelineLayout)
{

}

void vk::Scene::AddLightSource(Light& LightSource)
{
	m_Lights.push_back(std::move(LightSource));
}

void vk::Scene::Update(GLFWwindow* window)
{

	for (auto& light : m_Lights)
	{
		glm::mat4 ortho = glm::ortho(-11.0f, 11.0f, -11.0f, 11.0f, 01.f, 28.1f);
		glm::mat4 view = glm::lookAt(glm::vec3(light.position), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0));
		light.LightSpaceMatrix = ortho * view;
	}

	// Fill GPU Data with data defined for the scene
	for (size_t i = 0; i < m_Lights.size(); i++)
	{
		m_LightBuffer.lights[i].type = static_cast<int>(m_Lights[i].Type);
		m_LightBuffer.lights[i].LightPosition = m_Lights[i].position;
		m_LightBuffer.lights[i].LightColour = m_Lights[i].colour;
		m_LightBuffer.lights[i].LightSpaceMatrix = m_Lights[i].LightSpaceMatrix;
	}

	// Pass the light data to the GPU to update all light properties
	m_LightUBO[currentFrame].WriteToBuffer(m_LightBuffer, sizeof(LightBuffer));
}

void vk::Scene::Destroy()
{
	vertexBuffer.Destroy(context.device);
	indexBuffer.Destroy(context.device);
	meshOffsetBuffer.Destroy(context.device);
	RTMaterialsBuffer.Destroy(context.device);
	instanceBuffer->Destroy(context.device);
	vkDestroyAccelerationStructureKHR(context.device, TopLevelAccelerationStructure.handle, nullptr);
	TopLevelAccelerationStructure.buffer->Destroy(context.device);

	for (auto& BLAS : BottomLevelAccelerationStructures)
	{
		vkDestroyAccelerationStructureKHR(context.device, BLAS.handle, nullptr);
		BLAS.buffer->Destroy(context.device);
	}

	for (auto& texture : textures)
	{
		texture.Destroy(context.device);
	}
	// Destroy model resources for the GLTF resources loaded in
	if (!gltfModels.empty())
	{
		for (auto& model : gltfModels)
		{
			model.Destroy();
		}
	}

	for (auto& buffer : m_LightUBO)
	{
		buffer.Destroy(context.device);
	}
}
