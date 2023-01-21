#include "PCH.h"
#include "Application.h"

namespace Tigraf
{
	#define BIND_EVENT_FN(x) std::bind(&x, this, std::placeholders::_1)

	Application* Application::s_Instance = nullptr;

	Application::Application(ApplicationSpecification spec)
		: m_ApplicationSpecification(spec)
	{
		if(s_Instance != nullptr) delete s_Instance;
		s_Instance = this;
	}

	Application::~Application()
	{
		s_Instance = nullptr;

		for(Layer* layer : m_Layers) delete layer;
	}

	void Application::run()
	{
		init();

		while(m_Running)
		{
			onUpdate();
			onDraw();
		}

		shutdown();
	}

	void Application::init()
	{
		CORE_TRACE(getWorkingDirectory());

		m_Window = Window::createWindow(1280, 720, "Tigraf", false, BIND_EVENT_FN(Application::onEvent)); //Handles Window, Events and Graphics Context
		m_Timer = createScope<glfwTimer>();
		
		Renderer::init();
		for(Layer* layer : m_Layers) layer->init();
	}

	void Application::shutdown()
	{
		CORE_INFO("Shutting down");

		Renderer::shutdown();
		for(Layer* layer : m_Layers) layer->shutdown();

		delete this;
	}

	void Application::onUpdate()
	{
		//CORE_TRACE("Application::onUpdate");

		TimeStep ts = m_Timer->advance();

		m_Window->onUpdate(ts);

		for(Layer* layer : m_Layers) layer->onUpdate(ts);
	}

	void Application::onDraw()
	{
		//CORE_TRACE("Application::onDraw");
		Renderer::s_RendererAPI->clear();

		for(Layer* layer : m_Layers) layer->onDraw();
	}

	bool Application::onEvent(Event& event)
	{
		//CORE_TRACE("Application::onEvent");
		DISPATCH(EVENT_TYPE::RESIZE, event, Application::onResize);
		DISPATCH(EVENT_TYPE::CLOSE, event, Application::onClose);
		DISPATCH(EVENT_TYPE::KEY_PRESS, event, Application::onKey);

		for(Layer* layer : m_Layers) if(layer->onEvent(event)) break;
	}

	bool Application::onResize(void* eventData)
	{
		ResizeData* data = (ResizeData*)eventData;
		CORE_TRACE(*data);

		m_Minimized = !(data->width && data->height);

		if(!m_Minimized)
		{
			Renderer::s_RendererAPI->setViewport(0, 0, data->width, data->height);
			onDraw();
		}

		m_Window->onUpdate({});
		return m_Minimized;
	}

	bool Application::onClose(void* eventData)
	{
		CORE_TRACE("CloseEvent");
		m_Running = false;
		return true;
	}

	bool Application::onKey(void* eventData)
	{
		KeyData* data = (KeyData*)eventData;
		CORE_TRACE(*data);
		return false;
	}
}