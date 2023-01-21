#include "PCH.h"
#include "Window.h"

#include "glfwWindow.h"

namespace Tigraf
{
    Scope<Window> Tigraf::Window::createWindow(int width, int height, const char* name, bool enableVsync, std::function<void(Event&)> eventFunc)
    {        
        return createScope<glfwWindow>(width, height, name, enableVsync, eventFunc);
    }
}
