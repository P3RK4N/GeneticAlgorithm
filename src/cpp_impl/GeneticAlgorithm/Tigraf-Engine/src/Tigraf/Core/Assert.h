#pragma once

#include "TIGRAF/Core/Core.h"
#include "TIGRAF/Core/Log.h"
#include <filesystem>

#ifdef TIGRAF_ENABLE_ASSERTS
	#define TIGRAF_INTERNAL_ASSERT_IMPL(type, check, msg, ...) { if(!(check)) { TIGRAF##type##ERROR(msg, __VA_ARGS__); TIGRAF_DEBUGBREAK(); } }
	#define TIGRAF_INTERNAL_ASSERT_WITH_MSG(type, check, ...) TIGRAF_INTERNAL_ASSERT_IMPL(type, check, "Assertion failed: {0}", __VA_ARGS__)
	#define TIGRAF_INTERNAL_ASSERT_NO_MSG(type, check) TIGRAF_INTERNAL_ASSERT_IMPL(type, check, "Assertion '{0}' failed at {1}:{2}", TIGRAF_STRINGIFY_MACRO(check), std::filesystem::path(__FILE__).filename().string(), __LINE__)

	#define TIGRAF_INTERNAL_ASSERT_GET_MACRO_NAME(arg1, arg2, macro, ...) macro
	#define TIGRAF_INTERNAL_ASSERT_GET_MACRO(...) TIGRAF_EXPAND_MACRO( TIGRAF_INTERNAL_ASSERT_GET_MACRO_NAME(__VA_ARGS__, TIGRAF_INTERNAL_ASSERT_WITH_MSG, TIGRAF_INTERNAL_ASSERT_NO_MSG) )

	// Currently accepts at least the condition and one additional parameter (the message) being optional
	#define TIGRAF_ASSERT(...) TIGRAF_EXPAND_MACRO( TIGRAF_INTERNAL_ASSERT_GET_MACRO(__VA_ARGS__)(_, __VA_ARGS__) )
	#define TIGRAF_CORE_ASSERT(...) TIGRAF_EXPAND_MACRO( TIGRAF_INTERNAL_ASSERT_GET_MACRO(__VA_ARGS__)(_CORE_, __VA_ARGS__) )

#else
	#define TIGRAF_ASSERT(...)
	#define TIGRAF_CORE_ASSERT(...)

#endif

#ifdef TIGRAF_ENABLE_ASSERTS
	#define TIGRAF_ASSERT(x, ...) { if(!(x)) { ERROR("Assertion failed: {0}", __VA_ARGS__); TIGRAF_DEBUGBREAK(); } }
	#define TIGRAF_CORE_ASSERT(x, ...) { if(!(x)) { CORE_ERROR("Assertion failed: {0}", __VA_ARGS__); TIGRAF_DEBUGBREAK(); } }
#else
	#define TIGRAF_ASSERT(x, ...)
	#define TIGRAF_CORE_ASSERT(x, ...)
#endif 