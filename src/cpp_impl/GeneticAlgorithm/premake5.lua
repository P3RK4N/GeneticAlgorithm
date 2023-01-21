include "Dependencies.lua"

workspace "GeneticAlgorithm"
   architecture "x64"
   startproject "GeneticAlgorithm-ConsoleApp"
   configurations { "Debug", "Release", "Dist" }
   flags { "MultiProcessorCompile" }

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

group "Dependencies"
   include "vendor/premake"
   include "Tigraf-Engine/vendor/glfw"
   include "Tigraf-Engine/vendor/glad"
--     include "GCEngine/vendor/imgui"
--     include "GCEngine/vendor/yaml-cpp"
--     include "GCEngine/vendor/Box2D"
group ""

group "Core"
   include "GeneticAlgorithm-ConsoleApp"
   include "Tigraf-Engine"
group ""