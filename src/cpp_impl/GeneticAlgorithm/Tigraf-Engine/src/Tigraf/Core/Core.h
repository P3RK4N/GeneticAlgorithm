#pragma once
#include <memory>
#include "Tigraf/Core/PlatformDetection.h"

#ifdef TIGRAF_WINDOWS
	#if TIGRAF_DYNAMIC_LINK
		#ifdef TIGRAF_DLL
			#define TIGRAF_API __declspec(dllexport)
		#else 
			#define TIGRAF_API __declspec(dllimport)
		#endif
	#else 
		#define TIGRAF_API
	#endif
#else
	#error Tigraf only supports windows!
#endif

#ifdef TIGRAF_DEBUG

	#if defined(TIGRAF_WINDOWS)
		#define TIGRAF_DEBUGBREAK() __debugbreak()

	#elif defined(TIGRAF_LINUX)
		#include <signal.h>
		#define TIGRAF_DEBUGBREAK() raise(SIGTRAP)

	#else
		#error "Platform doesn't support debugbreak yet!"
	
	#endif
	
	#define TIGRAF_ENABLE_ASSERTS

#else
	#define TIGRAF_DEBUGBREAK()

#endif


#define TIGRAF_EXPAND_MACRO(x) x
#define TIGRAF_STRINGIFY_MACRO(x) #x
#define BIT(x) (1 << x)
#define BIND_EVENT_FN(fn) std::bind(&fn, this, std::placeholders::_1) 


#define DEL(ptr) if(ptr) delete ptr;

#define defaultConstructor(CLASS)								\
	private:													\
		CLASS() = default;										\
		CLASS(const CLASS&) = default;							\
		CLASS& operator=(const CLASS&) = default;	


namespace Tigraf
{
	template<typename T>
	using Scope = std::unique_ptr<T>;

	template<typename T, typename ... Args> 
	constexpr Scope<T> createScope(Args&& ... args) { return std::make_unique<T>(std::forward<Args>(args)...); }

	template<typename T>
	using Ref = std::shared_ptr<T>;

	template<typename T, typename ... Args>
	constexpr Ref<T> createRef(Args&& ... args) { return std::make_shared<T>(std::forward<Args>(args)...); }
}

#include "Tigraf/Core/Log.h"
#include "Tigraf/Core/Assert.h"