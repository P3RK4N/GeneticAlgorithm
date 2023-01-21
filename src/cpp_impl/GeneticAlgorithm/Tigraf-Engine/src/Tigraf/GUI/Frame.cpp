#include "PCH.h"
#include "Frame.h"

namespace Tigraf
{
	uint32_t Frame::s_PanelBarHeight = 20;
	
	Frame::Frame()
	{
		
	}

	Frame::~Frame()
	{

	}

	void Frame::onUpdate(const TimeStep& ts)
	{
	
	}

	void Frame::onDraw()
	{
	
	}

	bool Frame::onEvent(Event& e)
	{
		return false;
	}

	bool Frame::onMouseMove(void* eventData)
	{
		return false;
	}

	bool Frame::onButtonPress(void* eventData)
	{
		return false;

	}

	bool Frame::onMouseClick(void* eventData)
	{
		return false;
	}

	bool Frame::onRelease(void* eventData)
	{
		return false;
	}

	bool Frame::onResize(void* eventData)
	{
		return false;
	}

}