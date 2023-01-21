project "GeneticAlgorithm-ConsoleApp"
   kind "ConsoleApp"
   language "C++"
   cppdialect "C++20"
   staticruntime "on"
   
   targetdir("%{wks.location}/bin/" .. outputdir .. "/%{prj.name}")
   objdir("%{wks.location}/bin-int/" .. outputdir .. "/%{prj.name}")
   
   files
   {
      "src/**.h",
      "src/**.cpp"
   }
   
   includedirs
   {
      Includes["Tigraf-Engine"],
      Includes["spdlog"],
      Includes["stb"],
      Includes["glfw"],
      Includes["glad"],
      Includes["glm"]
   }

   links
   {
      "Tigraf-Engine"
   }

   defines
   {
   }

   filter "system:windows"
      systemversion "latest"
      defines
      {
      }

   filter "configurations:Debug"
      defines "TIGRAF_DEBUG"
      runtime "Debug"
      symbols "on"
      
   filter "configurations:Release"
      defines "TIGRAF_RELEASE"
      runtime "Release"
      optimize "on"

   filter "configurations:Dist"
      defines "TIGRAF_DIST"
      runtime "Release"
      optimize "on"