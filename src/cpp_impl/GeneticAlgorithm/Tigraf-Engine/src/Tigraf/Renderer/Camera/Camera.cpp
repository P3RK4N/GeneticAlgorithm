#include "PCH.h"
#include "Camera.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/transform.hpp>

namespace Tigraf
{
	void Camera::setAspectRatio(float aspectRatio)	
	{ 
		m_AspectRatio = aspectRatio; 
		recalculateProjection(); 
	}

	void Camera::setNearPlane(float nearPlane)		
	{ 
		m_NearPlane = nearPlane; 
		recalculateProjection(); 
	}

	void Camera::setFarPlane(float farPlane)		
	{ 
		m_FarPlane = farPlane; 
		recalculateProjection(); 
	}

	void Camera::setFOV(float FOV)
	{
		m_FOV = FOV;
		recalculateProjection();
	}

	void Camera::recalculateView()
	{
		m_View = glm::lookAt
		(
			m_Position,
			m_Position + m_Forward,
			glm::vec3(0,1,0)
		);
	}

	void Camera::recalculateProjection()
	{
		if(m_ProjectionType == ProjectionType::ORTHOGRAPHIC)
		{
			//m_Projection = glm::ortho();
		}
		else
		{
			m_Projection = glm::perspective(m_FOV, m_AspectRatio, m_NearPlane, m_FarPlane);
		}
	}


}