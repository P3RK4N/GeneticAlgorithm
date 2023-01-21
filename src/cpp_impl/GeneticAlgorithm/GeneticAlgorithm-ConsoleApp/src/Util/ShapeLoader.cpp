#include "ShapeLoader.h"

#include <Tigraf/Core/Core.h>

#include <fstream>
#include <iostream>

Shape ShapeLoader::LoadShape(const std::filesystem::path& rootFolder)
{
	std::fstream points(rootFolder/"points.txt", std::ios::in);
	std::fstream muscles(rootFolder/"muscles.txt", std::ios::in);
	std::fstream bones(rootFolder/"bones.txt", std::ios::in);

	Shape shape{};
	std::string line;

	while(points && std::getline(points, line))
	{
		int delimiter1 = line.find_first_of(" ");
		int delimiter2 = line.find_last_of(" ");

		float x = std::stof(line.substr(0, delimiter1));
		float y = std::stof(line.substr(delimiter1+1, delimiter2));
		float z = std::stof(line.substr(delimiter2 + 1));

		shape.Points.emplace_back(x, y, z);
	}

	TIGRAF_ASSERT(shape.Points.size() > 0 && shape.Points.size() <= MAX_POINTS, "Invalid amount of points");

	while(muscles && std::getline(muscles, line))
	{
		int delimiter = line.find_first_of(" ");
		int x = std::stoi(line.substr(0, delimiter));
		int y = std::stoi(line.substr(delimiter+1));

		shape.Muscles.emplace_back(x, y);
		shape.MusclesLength.emplace_back(glm::distance(shape.Points[x], shape.Points[y]));
	}

	TIGRAF_ASSERT(shape.Muscles.size() >= 0 && shape.Muscles.size() <= MAX_MUSCLES, "Invalid amount of muscles");

	while(bones && std::getline(bones, line))
	{
		int delimiter = line.find_first_of(" ");
		int x = std::stoi(line.substr(0, delimiter));
		int y = std::stoi(line.substr(delimiter+1));

		shape.Bones.emplace_back(x, y);
		shape.BoneLengths.emplace_back(glm::distance(shape.Points[x], shape.Points[y]));
	}

	TIGRAF_ASSERT(shape.Bones.size() >= 0 && shape.Bones.size() <= MAX_BONES, "Invalid amount of bones");

	return shape;
}

void ShapeLoader::PrintShape(const Shape& shape)
{
	std::cout << "\n";

	for(int i = 0; i < shape.Points.size(); i++) 
		std::cout << "Point " << i << ": " <<
		shape.Points[i].x << " " <<
		shape.Points[i].y << " " <<
		shape.Points[i].z << "\n";
		

	for(int i = 0; i < shape.Muscles.size(); i++)
		std::cout << "Muscle " << i << ": " <<
		shape.Muscles[i].x << " " <<
		shape.Muscles[i].y << "\n";

	for(int i = 0; i < shape.Bones.size(); i++)
		std::cout << "Bone " << i << ": " <<
		shape.Bones[i].x << " " <<
		shape.Bones[i].y << "\n";

}
