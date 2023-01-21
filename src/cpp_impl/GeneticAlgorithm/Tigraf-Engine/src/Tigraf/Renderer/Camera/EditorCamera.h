#pragma once
#include "Camera.h"

namespace Tigraf
{
	class EditorCamera : public Camera
	{
	public:
		EditorCamera
		(
			float aspectRatio, 
			float nearPlane, 
			float farPlane, 
			const glm::vec3& position = { 5, 1, -5 },
			const glm::vec3& forward = { 0, 0, 1 }, 
			const glm::vec3& up = { 0, 1, 0 },
			float FOV = 45.0f
		);

		virtual ~EditorCamera();

		virtual void onUpdate(const TimeStep& ts) override;

	private:
		void updateTransform(const TimeStep& ts);

		float m_MoveSpeed = 10.0f;
		float m_RotateSpeed = 20.0f;

		int m_CursorX = 0;
		int m_CursorY = 0;
	};
}
