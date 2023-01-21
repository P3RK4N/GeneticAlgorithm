#pragma once
#include "Shape.h"

#include <filesystem>

class ShapeLoader
{
public:
	static Shape LoadShape(const std::filesystem::path& rootFolder);
	static void PrintShape(const Shape& shape);
};