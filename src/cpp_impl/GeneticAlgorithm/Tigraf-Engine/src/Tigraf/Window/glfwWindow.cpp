#include "PCH.h"
#include "glfwWindow.h"

namespace Tigraf
{
	static int status = 0;

	glfwWindow::glfwWindow(int width, int height, const char* name, bool vsyncEnabled, std::function<void(Event&)> eventCallback)
	{
		status = glfwInit();
		TIGRAF_ASSERT(status, "GLFW library couldn't be initialized!");
		glfwSetErrorCallback(glfwWindow::onError);

		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);

		GLFWwindow* window = glfwCreateWindow(width, height, name, NULL, NULL);
		TIGRAF_ASSERT(window, "Window creation failed!");

		m_GraphicsContext.init = [window]()
		{
			glfwMakeContextCurrent(window);
			status = gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
			TIGRAF_ASSERT(status, "Could not initialize graphics context!");

			//-------------
			CORE_INFO("GRAPHICS CONTEXT");
			CORE_INFO("\tVendor: {0}", glGetString(GL_VENDOR));
			CORE_INFO("\tRenderer: {0}", glGetString(GL_RENDERER));
			CORE_INFO("\tVersion: {0}\n", glGetString(GL_VERSION));
			//-------------

			//------------------------
			CORE_INFO("UNIFORM BUFFERS:");
			
			glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &status);
			CORE_INFO("\tMAX UNIFORM BLOCK SIZE: {0}", status);

			glGetIntegerv(GL_MAX_COMBINED_UNIFORM_BLOCKS, &status);
			CORE_INFO("\tMAX COMBINED UNIFORM BLOCKS: {0}\n", status);
			//-------------------------

			//-------------------------
			CORE_INFO("TEXTURES:");

			glGetIntegerv(GL_MAX_TEXTURE_SIZE, &status);
			CORE_INFO("\tMAX TEXTURE SIZE: {0}x{0}", status);

			glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &status);
			CORE_INFO("\tMAX COMBINED TEXTURE IMAGE UNITS: {0}\n", status);
			//--------------------------
		};
		m_GraphicsContext.swapBuffers = [window]()
		{
			glfwSwapBuffers(window);
		};

		m_GraphicsContext.init();
		glfwSwapInterval(vsyncEnabled);

		this->m_WindowHandle = window;
		this->m_WindowData.m_Vsync = vsyncEnabled;
		this->m_WindowData.m_EventCallback = eventCallback;

		glfwSetWindowUserPointer(m_WindowHandle, &m_WindowData);

		setEventCallbacks();
	}

	glfwWindow::~glfwWindow()
	{
		glfwDestroyWindow(m_WindowHandle);
		m_WindowHandle = nullptr;
		glfwTerminate();
	}

	std::pair<int,int> glfwWindow::getSize()
	{
		int width, height;
		glfwGetWindowSize(m_WindowHandle, &width, &height);
		return { width,height };
	}

	void glfwWindow::onError(int error, const char* description)
	{
		fprintf(stderr, "Error: %s\n", description);
	}

	void glfwWindow::onUpdate(const TimeStep& ts)
	{
		static float timer = 0.0f;
		static float frames = 0;

		timer += ts;
		frames += (bool)(ts > 0.0f);

		if(timer > 1.0f)
		{
			std::string title = std::format("Tigraf - {}FPS / {}s", frames, 1.0f/frames);
			timer -= 1.0f;
			frames = 0;
			glfwSetWindowTitle(m_WindowHandle, title.c_str());
		}

		m_GraphicsContext.swapBuffers(); 
		glfwPollEvents();
	}

	void glfwWindow::setEventCallbacks()
	{
		glfwSetWindowSizeCallback(m_WindowHandle, [](GLFWwindow* window, int width, int height)
		{
			ResizeData data{ width, height };
			Event e{EVENT_TYPE::RESIZE, (void*)&data};
			
			WindowData* wd = (WindowData*)glfwGetWindowUserPointer(window);
			wd->m_EventCallback(e);
		});

		glfwSetWindowCloseCallback(m_WindowHandle, [](GLFWwindow* window)
		{
			Event e{EVENT_TYPE::CLOSE, nullptr};
			
			WindowData* wd = (WindowData*)glfwGetWindowUserPointer(window);
			wd->m_EventCallback(e);
		});

		glfwSetKeyCallback(m_WindowHandle, [](GLFWwindow* window, int key, int scancode, int action, int mods)
		{
			EVENT_TYPE et;

			switch(action)
			{
			case GLFW_PRESS:
				et = EVENT_TYPE::KEY_PRESS;
				break;
			case GLFW_RELEASE:
				et = EVENT_TYPE::KEY_RELEASE;
				break;
			case GLFW_REPEAT:
				et = EVENT_TYPE::KEY_REPEAT;
			}

			KeyData data{ key };
			Event e{ et, (void*)&data };
			WindowData* wd = (WindowData*)glfwGetWindowUserPointer(window);
			wd->m_EventCallback(e);
		});

		glfwSetCursorPosCallback(m_WindowHandle, [](GLFWwindow* window, double xPos, double yPos)
		{
			CursorData data{ (int)xPos, (int)yPos };
			Event e{ EVENT_TYPE::CURSOR_MOVE, (void*)&data };
			WindowData* wd = (WindowData*)glfwGetWindowUserPointer(window);
			wd->m_EventCallback(e);
		});
	}
}