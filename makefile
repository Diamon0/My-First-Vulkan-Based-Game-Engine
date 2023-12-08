CFLAGS = -std=c++20 -O2

LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

.PHONY: Debug Test Clean

Debug: Source.cpp
	g++ $(CFLAGS) -o VulkanGameTest Source.cpp $(LDFLAGS)

Test: VulkanTest
	./VulkanGameTest

Clean:
	rm -f VulkanGameTest
