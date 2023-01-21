#include "PCH.h"
#include "GUI_Element.h"

namespace Tigraf
{
	GUI_Element* GUI_Element::s_Active	= nullptr;
	GUI_Element* GUI_Element::s_Hovered	= nullptr;
	GUI_Element* GUI_Element::s_Held	= nullptr;
	GUI_Element* GUI_Element::s_Dragged	= nullptr;
}