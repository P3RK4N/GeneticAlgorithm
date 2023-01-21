#include "PCH.h"
#include "EditorCamera.h"

#include "Tigraf/Input/glfwInput.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/transform.hpp>

namespace Tigraf
{
	EditorCamera::EditorCamera
	(
		float aspectRatio, 
		float nearPlane, 
		float farPlane, 
		const glm::vec3& position,
		const glm::vec3& forward,
		const glm::vec3& up,
		float FOV
	)
		: Camera(ProjectionType::PERSPECTIVE, aspectRatio, nearPlane, farPlane, position, forward, up, FOV)
	{
		auto[x, y] = glfwInput::getCursorPos();
		m_CursorX = x;
		m_CursorY = y;

		recalculateView();
		recalculateProjection();
	}

	EditorCamera::~EditorCamera()
	{
	
	}

	void EditorCamera::onUpdate(const TimeStep& ts)
	{
		updateTransform(ts);

		recalculateView();
		recalculateViewProjection();
	}

	void EditorCamera::updateTransform(const TimeStep& ts)
	{
		float moveFactor = ts * m_MoveSpeed;

		if(glfwInput::isKeyDown(KEY_LEFT_CONTROL)) moveFactor *= 5.0f;

		glm::vec3 forwardOffset = m_Forward * moveFactor;
		glm::vec3 sideVector = glm::normalize(glm::cross(m_Forward, { 0, 1, 0 }));
		glm::vec3 sideOffset = sideVector * moveFactor;
		glm::vec3 upOffset{ 0,moveFactor,0 };

		if(glfwInput::isKeyDown(KEY_A)) m_Position -= sideOffset;
		if(glfwInput::isKeyDown(KEY_D)) m_Position += sideOffset;
		if(glfwInput::isKeyDown(KEY_S)) m_Position -= forwardOffset;
		if(glfwInput::isKeyDown(KEY_W)) m_Position += forwardOffset;
		if(glfwInput::isKeyDown(KEY_SPACE)) m_Position += upOffset;
		if(glfwInput::isKeyDown(KEY_LEFT_SHIFT)) m_Position -= upOffset;

		auto [x, y] = glfwInput::getCursorPos();
		int dx = x - m_CursorX;
		int dy = y - m_CursorY;
		m_CursorX = x;
		m_CursorY = y;

		if(glfwInput::isButtonDown(MOUSE_BUTTON_LEFT))
		{
			float rotateFactor = ts * m_RotateSpeed;

			glm::mat4 rotationMat{ 1.0f };
			rotationMat = glm::rotate(glm::radians(-dy * rotateFactor), sideVector) * glm::rotate(glm::radians(-dx * rotateFactor), glm::vec3{ 0,1,0 });

			m_Forward = glm::vec3(rotationMat * glm::vec4(m_Forward, 1.0f));
			m_Forward = glm::normalize(m_Forward);
		
			m_Up = glm::vec3(rotationMat * glm::vec4(m_Up, 1.0f));
			m_Up = glm::normalize(m_Up);
		}
	}

}