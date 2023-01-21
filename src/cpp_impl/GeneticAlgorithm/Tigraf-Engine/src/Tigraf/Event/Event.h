#pragma once
#include <string>

namespace Tigraf
{
#define DISPATCH(TYPE, EVENT, FUNC) if(EVENT.m_Type == TYPE && FUNC(EVENT.m_Data)) return true

#define EVENT(NAME) bool NAME(void* eventData)

	enum class EVENT_TYPE : int
	{
		RESIZE,
		CURSOR_MOVE,
		MINIMIZE,
		CLOSE,
		KEY_PRESS,
		KEY_RELEASE,
		KEY_REPEAT
	};

	class Event
	{
	public:
		Event(EVENT_TYPE type, void* data)
			: m_Type(type), m_Data(data) {}

		EVENT_TYPE m_Type;
		void* m_Data;
	};

	struct ResizeData
	{
		ResizeData(int width, int height) : width(width), height(height) {}
		int width = 0;
		int height = 0;

		const std::string ToString() const { return std::format("ResizeEvent({0},{1})", width, height); }
	};

	struct KeyData
	{
		KeyData(int keycode) : keycode(keycode) {}
		int keycode = 0;

		const std::string ToString() const { return std::format("KeyEvent({0})", keycode); }
	};

	struct CursorData
	{
		CursorData(int xPos, int yPos) : xPos(xPos), yPos(yPos) {}
		int xPos = 0;
		int yPos = 0;

		const std::string ToString() const { return std::format("CursorEvent({0},{1})", xPos, yPos); }
	};

#define CAST_STREAM(DATA) inline std::ostream& operator<<(std::ostream& os, const DATA& d) { return os << d.ToString(); }

	CAST_STREAM(ResizeData)
	CAST_STREAM(KeyData)
	CAST_STREAM(CursorData)
}