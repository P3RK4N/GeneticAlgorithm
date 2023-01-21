#pragma once
#include "Tigraf/Core/Core.h"
#include "GUI_Element.h"

namespace Tigraf
{
	class Frame : public GUI_Element
	{
		DECLARE_GUI_ELEMENT

	public:
		Frame();
		virtual ~Frame() override;

	protected:
		static uint32_t s_PanelBarHeight;

	protected:
		std::vector<Ref<Frame>> TopFrames{};
		std::vector<Ref<Frame>> RightFrames{};
		std::vector<Ref<Frame>> BottomFrames{};
		std::vector<Ref<Frame>> LeftFrames{};

		friend class Panel;
	};
}