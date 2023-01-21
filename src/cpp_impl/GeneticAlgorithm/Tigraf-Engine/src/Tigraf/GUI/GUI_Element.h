#pragma once
#include "Tigraf/Event/Event.h"
#include "Tigraf/Core/Timer.h"

namespace Tigraf
{
#define DECLARE_GUI_ELEMENT								\
protected:												\
	virtual void onUpdate(const TimeStep& ts) override;	\
	virtual void onDraw() override;						\
	virtual bool onEvent(Event& e) override;			\
	virtual EVENT(onMouseMove) 	 override;				\
	virtual EVENT(onResize) 	 override;				\
	virtual EVENT(onMouseClick)  override;				\
	virtual EVENT(onButtonPress) override;				\
	virtual EVENT(onRelease) 	 override;	


	class GUI_Element
	{
	public:
		virtual ~GUI_Element() {}

		bool isHovered()	{ return this == s_Hovered; }
		bool isHeld()		{ return this == s_Held;	}
		bool isDragged()	{ return this == s_Dragged; }
		bool isActive()		{ return this == s_Active;	}

	protected:
		static GUI_Element* s_Dragged;
		static GUI_Element* s_Held;
		static GUI_Element* s_Hovered;
		static GUI_Element* s_Active;

	protected:
		virtual void onUpdate(const TimeStep& ts) = 0;
		virtual void onDraw() = 0;
		virtual bool onEvent(Event& e) = 0;

		virtual EVENT(onMouseMove) = 0;
		virtual EVENT(onResize) = 0;
		virtual EVENT(onMouseClick) = 0;
		virtual EVENT(onButtonPress) = 0;
		virtual EVENT(onRelease) = 0;

	protected:
		bool m_Held = false;
		bool m_Hovered = false;
		bool m_Dragged = false;

		uint32_t m_TopLeft = 0;
		uint32_t m_BottomRight = 0;
	};
}