#pragma once
#include "Tigraf/Core/Timer.h"

#include <glm/glm.hpp>

namespace Tigraf
{
	enum class ProjectionType : int
	{
		ORTHOGRAPHIC,
		PERSPECTIVE
	};

	class Camera
	{
	public:
		Camera
		(
			ProjectionType type,
			float aspectRatio, 
			float nearPlane, 
			float farPlane, 
			const glm::vec3& position = { 0, 1, -3 }, 
			const glm::vec3& forward = { 0, 0, 1 },
			const glm::vec3& up = { 0, 1, 0 },
			float FOV = 45.0f
		)
			: m_AspectRatio(aspectRatio), 
			m_NearPlane(nearPlane), 
			m_FarPlane(farPlane), 
			m_Position(position), 
			m_Forward(forward),
			m_Up(up),
			m_FOV(FOV),
			m_ProjectionType(type) {}

		virtual ~Camera() {}

		virtual void onUpdate(const TimeStep& ts) = 0;

		const glm::mat4& getProjection() { return m_Projection; }
		const glm::mat4& getView() { return m_View; }
		const glm::mat4& getViewProjection() { return m_ViewProjection; }

		const glm::vec3& getForward() { return m_Forward; }
		const glm::vec3& getPosition() { return m_Position; }

		void setForward(const glm::vec3& forward) { m_Forward = forward; }
		void setPosition(const glm::vec3& position) { m_Position = position; }

		const float getAspectRatio() { return m_AspectRatio; }
		const float getNearPlane() { return m_NearPlane; }
		const float getFarPlane() { return m_FarPlane; }
		const float getFOV() { return m_FOV; }

		void setAspectRatio(float aspectRatio);
		void setNearPlane(float nearPlane);
		void setFarPlane(float farPlane);
		void setFOV(float FOV);

		void recalculateView();
		void recalculateProjection();
		void recalculateViewProjection() { m_ViewProjection = m_Projection * m_View; }

	protected:
		ProjectionType m_ProjectionType;
		glm::mat4 m_ViewProjection = {};
		glm::mat4 m_Projection = {};
		glm::mat4 m_View = {};

		glm::vec3 m_Forward = {};
		glm::vec3 m_Up = {};

		glm::vec3 m_Position = {};

		float m_AspectRatio = 1280.0f / 720.0f;
		float m_NearPlane = 0.01f;
		float m_FarPlane = 100.0f;
		float m_FOV = 45.0f;

	};
}