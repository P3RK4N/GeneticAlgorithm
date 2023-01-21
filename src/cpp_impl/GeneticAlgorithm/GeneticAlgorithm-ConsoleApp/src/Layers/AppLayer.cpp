#include "AppLayer.h"

#include "../Util/ShapeLoader.h"
#include "../Util/ShapeRenderer.h"

#include <Tigraf/Input/Keycodes.h>
#include <Tigraf/Input/glfwInput.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/transform.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


#define DRAW_X 128
#define DRAW_Y 128

//32x32		=> 1		
//64x64		=> 4
//128x128	=> 16
//256x256	=> 64
//512x512	=> 256
//1024x1024	=> 1024
#define COMPUTE_X 16
#define COMPUTE_Y 1
#define COMPUTE_Z 1

namespace Tigraf
{
	struct PerFrameData
	{
		glm::mat4 CameraViewProjection{};
		glm::vec3 CameraWorldPosition{};
		float TotalTime;
		float FrameTime;
	} frameData{};

	struct PerModelData
	{
		glm::mat4 M{};
		glm::mat4 MVP{};
	} modelData{};

	void AppLayer::init()
	{
		//SHAPE
#if SPIDER
		Shape shape = ShapeLoader::LoadShape("resources\\shapes\\spider");
#else
		Shape shape = ShapeLoader::LoadShape("resources\\shapes\\biped");
#endif

		ShapeRenderer::InitShaders();

		ShapeRenderer::FillBuffers(shape, RW_BUFFER_0, RW_BUFFER_2, RW_BUFFER_4, RW_BUFFER_7, RW_BUFFER_5, RW_BUFFER_6, UNIFORM_BUFFER_3, DRAW_X*DRAW_Y);

		//FRAMEBUFFER FRAME
		glm::mat4 transform =  glm::scale(glm::vec3(2.0f, 2.0f, 2.0f));
		m_FramebufferFrameMesh = MeshPrimitives::Plane(transform);
		m_FramebufferFrameMesh->setShader(Shader::create("resources\\shaders\\FramebufferShader.glsl"));

		//FLOOR
		transform = glm::scale(glm::vec3(1000.0f, 1000.0f, 1000.0f)) * glm::mat4(1.0f);
		m_FloorMesh = MeshPrimitives::Plane(transform);
		m_FloorMesh->setShader(Shader::create("resources\\shaders\\GridShader.glsl"));

		//CUBE
		m_CubemapMesh = MeshPrimitives::Cube(transform);
		m_CubemapMesh->setShader(Shader::create("resources\\shaders\\CubemapShader.glsl"));

		//EDITOR_CAMERA
		auto[x, y] = Application::s_Instance->getWindow()->getSize();
		m_EditorCamera = createRef<EditorCamera>(1.0f * x / y, 0.1f, 1000.0f);

		//TEXTURES
		m_CubemapTexture = TextureCube::create("resources\\textures\\cubemaps\\skybox\\skybox", "jpg");
		SET_TEXTURE_HANDLE(m_CubemapTexture->getTextureHandle(), TEXTURE_CUBE_0);

		//FRAMEBUFFER
		auto [width, height] = Application::s_Instance->getWindow()->getSize();
		m_Framebuffer = Framebuffer::create(width, height);
		m_Framebuffer->attachColorTexture(TextureFormat::RGBA8);
		m_Framebuffer->attachDepthStencilTexture(TextureFormat::DEPTH24STENCIL8);
		m_Framebuffer->invalidate();
		SET_TEXTURE_HANDLE(m_Framebuffer->getColorTexture(0)->getTextureHandle(), TEXTURE_2D_0);
	}

	void AppLayer::onUpdate(const TimeStep& ts)
	{
		m_EditorCamera->onUpdate(ts);

		auto [w, h] = Application::s_Instance->getWindow()->getSize();
		if(w != m_Framebuffer->getWidth() || h != m_Framebuffer->getHeight())
		{
			m_Framebuffer->resize(w, h);
			SET_TEXTURE_HANDLE(m_Framebuffer->getColorTexture(0)->getTextureHandle(), TEXTURE_2D_0);
		}

		frameData.CameraWorldPosition = m_EditorCamera->getPosition();
		frameData.CameraViewProjection = m_EditorCamera->getViewProjection();
		frameData.TotalTime = ts.m_TotalTime;
		frameData.FrameTime = ts.m_FrameTime;
		UPDATE_PER_FRAME_BUFFER(frameData, sizeof(PerFrameData), 0);

		ShapeRenderer::ComputeMovement(COMPUTE_X, COMPUTE_Y, COMPUTE_Z);

		ShapeRenderer::ComputePhysics(COMPUTE_X, COMPUTE_Y, COMPUTE_Z);
	}

	void AppLayer::onDraw()
	{
		m_Framebuffer->bind();
		{
			m_CubemapMesh->drawTrianglesIndexed();
			m_FloorMesh->drawTrianglesIndexed();

			static int drawAmount = 1;
			if(glfwInput::isKeyDown(KEY_LEFT)) 
			{
				drawAmount = std::max(1, drawAmount - 32);
			}
			if(glfwInput::isKeyDown(KEY_RIGHT)) 
			{
				drawAmount = std::min((int)(DRAW_X*DRAW_Y), drawAmount + 32);
			}
			ShapeRenderer::DrawInstances(drawAmount);
		}
		m_Framebuffer->unbind();

		m_FramebufferFrameMesh->drawTrianglesIndexed();
	}

	void AppLayer::shutdown()
	{
	
	}

	bool AppLayer::onEvent(Event& event)
	{
		DISPATCH(EVENT_TYPE::RESIZE, event, onResize);
	}

	bool AppLayer::onResize(void* eventData)
	{
		ResizeData* data = (ResizeData*)eventData; 

		m_EditorCamera->setAspectRatio(1.0f * data->width / data->height);
		m_EditorCamera->recalculateViewProjection();

		return false; 
	}
}