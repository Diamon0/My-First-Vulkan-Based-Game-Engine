@echo off
set /p shaderName=What is the name of the .vert and .frag pair (exclude extensions)? 
echo Compiling %shaderName%.vert into vert.spv and %shaderName%.frag into frag.spv
@echo on
C:\VulkanSDK\1.3.268.0\Bin\glslc.exe %shaderName%.vert -o vert.spv
C:\VulkanSDK\1.3.268.0\Bin\glslc.exe %shaderName%.frag -o frag.spv
@echo off
echo Finished
pause