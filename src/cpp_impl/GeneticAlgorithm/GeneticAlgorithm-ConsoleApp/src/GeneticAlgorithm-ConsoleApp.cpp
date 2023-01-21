#include <Tigraf.h>

#include "Layers/AppLayer.h"

using namespace Tigraf;

//Add layers to Application
void STARTUP(std::vector<Layer*>& layers)
{
	layers.push_back(new AppLayer());
}