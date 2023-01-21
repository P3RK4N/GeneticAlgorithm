#pragma once

namespace Tigraf
{
	class Input
	{
		static bool isKeyDown(int keycode) { return false; }
		static bool isKeyUp(int keycode) { return false; }

		//static std::pair<double, double> getMouseDelta() { return { 0.,0. }; }
	};
}
