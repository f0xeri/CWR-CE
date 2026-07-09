file(GLOB_RECURSE POSEIDON_VULKAN_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

set(POSEIDON_VULKAN_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/EngineVulkan.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/VulkanContext.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/VulkanSwapchain.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/VulkanShaderCompiler.hpp
)

add_library(PoseidonVulkan STATIC
    ${POSEIDON_VULKAN_SOURCES}
    ${POSEIDON_VULKAN_HEADERS}
)

find_package(VulkanHeaders CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(glslang CONFIG REQUIRED)

target_link_libraries(PoseidonVulkan PUBLIC Poseidon)
target_link_libraries(PoseidonVulkan PRIVATE
    Vulkan::Headers
    GPUOpen::VulkanMemoryAllocator
    glslang::glslang
    glslang::glslang-default-resource-limits
    glslang::SPIRV
)

# Vulkan-Hpp dynamic dispatch
target_compile_definitions(PoseidonVulkan PRIVATE
    VK_NO_PROTOTYPES
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
)

target_compile_definitions(PoseidonVulkan PRIVATE
    $<$<CONFIG:Debug>:_DEBUG>
)
if(WIN32)
    target_compile_definitions(PoseidonVulkan PRIVATE
        __MSC__=1100
        _WINDOWS
    )
endif()

if(MSVC OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"))
    target_compile_options(PoseidonVulkan PRIVATE
        /GR /Oi /GF
        $<$<CONFIG:Debug>:/Od>
        $<$<CONFIG:Debug>:/Ob1>
        $<$<CONFIG:Debug>:/MDd>
        /permissive
        /wd4291 /wd5030 /wd4799 /wd4068 /wd4996
        /wd4244 /wd4146 /wd4018 /wd4101 /wd4305 /wd4838
        $<$<CXX_COMPILER_ID:Clang>:-Wno-error=non-pod-varargs>
        $<$<CXX_COMPILER_ID:Clang>:-Wno-error=c++11-narrowing>
        $<$<CXX_COMPILER_ID:Clang>:-Wno-error>
    )
else()
    target_compile_options(PoseidonVulkan PRIVATE
        -Wno-error
        -Wno-c++11-narrowing
        -Wno-non-pod-varargs
        -Wno-missing-field-initializers
    )
endif()

set_target_properties(PoseidonVulkan PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "PoseidonVulkan"
)
