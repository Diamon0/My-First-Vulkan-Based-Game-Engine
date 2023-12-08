# My First Vulkan Game Engine! (WIP)

This is a project I am making both as a learning experience and as a goal for my biggest milestone yet in my game dev career!

While I could simply use an existing game engine (UE5, Godot, GameMaker, etc...), I believe that I won't be learning anought about what's happening under the hood, which ties into my reason for using Vulkan; and more so, Vulkan over OpenGL or DirectX.

- - -

# TODOs:

+ Refine the rateSuitability function: Right now it's extremely basic and only made for development.
+ Separate Files: Right now everything is in a single, all-powerful file and I don't enjoy it.
+ Organize The Code: Ties in to the previous point.
+ Make my own Assets: Like, I am still using the assets I was given, except for the icon which I have only used for the visual studio setup project (I make .msi files for my friends to test on their different Windows machines).
+ Finish The Debug Window: I am using ImGui for that, and honestly this will probable be an ever-extending task I will only finish until after the game is done; this task refers to its current "DemoWindow" however.
+ Finish The Debug Window But For Real This Time (FTDWBFRTT).
+ Start Adding APIs To Tie Game Logic To Vulkan.

- - -

# Building

Link the necessary dependencies, I recommend using Vcpkg for this if you are on Windows just because of the convenience and Visual Studio tying well with it. Additionally, Vcpkg is primarily what the project uses anyways, so you'd only have to add the installed folder path to the project config.

That's about it, remember to compile the shaders whenever you change them or add new ones, in case someone for whatever reason uses this project.

- - -

# FAQ


*Will you be adding Linux support to the dev environment and the release target?*
Probably after I'm done with it, but right now I'm focused on the Windows dev environment and release.
