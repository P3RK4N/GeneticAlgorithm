#pragma once
#include "Tigraf/Core/Core.h"
#include "Tigraf/Core/Application.h"

#include <iostream>

#ifdef TIGRAF_WINDOWS
#include <windows.h>

using namespace Tigraf;

extern void STARTUP(std::vector<Layer*>& layers);

int main(int argc, char** argv)
{
	Log::init();
	CORE_INFO("  Tiny Graphic Framework!");

	auto app = new Application(ApplicationSpecification{ "TIny GRAphic Framework", std::filesystem::current_path(), { argc, argv } });
	TIGRAF_ASSERT(app, "App creation was unsuccessful");

	STARTUP(app->getLayers());

	app->run();

	CORE_INFO("Application has successfully shut down!");
	std::cin.get();
}

#endif