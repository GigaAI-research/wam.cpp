cmake_minimum_required(VERSION 3.22)

project(wam VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(WamOptions)
include(Dependencies)
include(WamModels)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(WAM_PROTOC_EXECUTABLE)
    if(NOT EXISTS "${WAM_PROTOC_EXECUTABLE}")
        message(FATAL_ERROR
            "WAM_PROTOC_EXECUTABLE does not exist: ${WAM_PROTOC_EXECUTABLE}")
    endif()
else()
    unset(WAM_PROTOC_EXECUTABLE CACHE)
    find_program(WAM_PROTOC_EXECUTABLE protoc
        DOC "Optional protoc executable used to package wam.desc")
endif()

set(WAM_PROTO_DESCRIPTOR_AVAILABLE FALSE)
if(WAM_PROTOC_EXECUTABLE)
    set(WAM_PROTO_DESCRIPTOR "${CMAKE_CURRENT_BINARY_DIR}/wam.desc")
    add_custom_command(
        OUTPUT "${WAM_PROTO_DESCRIPTOR}"
        COMMAND "${WAM_PROTOC_EXECUTABLE}"
            "--proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto"
            "--descriptor_set_out=${WAM_PROTO_DESCRIPTOR}"
            wam.proto
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/proto/wam.proto"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/proto"
        COMMENT "Generating wam.desc from proto/wam.proto"
        VERBATIM)
    add_custom_target(wam-proto-descriptor ALL DEPENDS "${WAM_PROTO_DESCRIPTOR}")
    set(WAM_PROTO_DESCRIPTOR_AVAILABLE TRUE)
    message(STATUS "wam: Proto descriptor enabled with ${WAM_PROTOC_EXECUTABLE}")
else()
    message(STATUS
        "wam: protoc not found; core/CLI remain available, Python client and GigaWorld-Policy integration are unavailable")
endif()

wam_configure_dependencies()

add_library(wam_core
    src/model.cpp
    src/model_registry.cpp
)
add_library(wam::core ALIAS wam_core)
wam_add_model_sources(wam_core)

target_compile_features(wam_core PUBLIC cxx_std_17)
target_include_directories(wam_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(wam_core PRIVATE $<BUILD_INTERFACE:ggml>)
if(TARGET ggml-cpu)
    target_link_libraries(wam_core PRIVATE $<BUILD_INTERFACE:ggml-cpu>)
endif()
target_compile_options(wam_core PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-Wall;-Wextra;-Wpedantic>)
target_compile_definitions(wam_core PRIVATE
    WAM_RUNTIME_VERSION="${PROJECT_VERSION}"
    WAM_LLAMA_REVISION="${WAM_LLAMA_REVISION}"
    WAM_HAS_CUDA=$<BOOL:${WAM_CUDA}>
)
set_target_properties(wam_core PROPERTIES
    OUTPUT_NAME wam_core
    EXPORT_NAME core
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)

add_library(wam_c_api
    src/serving/c_api.cpp
    src/serving/protocol_adapter.cpp)
add_library(wam::c_api ALIAS wam_c_api)
target_link_libraries(wam_c_api PRIVATE wam::core)
target_include_directories(wam_c_api
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/serving)
target_compile_definitions(wam_c_api PRIVATE WAM_C_API_BUILD=1)
target_compile_options(wam_c_api PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
set_target_properties(wam_c_api PROPERTIES
    OUTPUT_NAME wam_c_api EXPORT_NAME c_api VERSION ${PROJECT_VERSION} SOVERSION 1)

if(WAM_BUILD_APPS)
    add_library(wam_app_support STATIC
        apps/replay.cpp
        ${llama_SOURCE_DIR}/examples/gguf-hash/deps/sha256/sha256.c)
    target_link_libraries(wam_app_support PRIVATE wam::core)
    target_include_directories(wam_app_support PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/apps
        ${llama_SOURCE_DIR}/vendor/nlohmann
        ${llama_SOURCE_DIR}/examples/gguf-hash/deps/sha256
        ${llama_SOURCE_DIR}/examples/gguf-hash/deps)
    target_compile_definitions(wam_app_support PRIVATE WAM_APP_CUDA=$<BOOL:${WAM_CUDA}>)
    target_compile_options(wam_app_support PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
    if(WAM_CUDA)
        target_link_libraries(wam_app_support PRIVATE CUDA::cudart)
    endif()

    add_executable(wam-cli apps/wam-cli.cpp)
    add_executable(wam-bench apps/wam-bench.cpp)
    add_executable(wam-compare apps/wam-compare.cpp)
    foreach(target IN ITEMS wam-cli wam-bench wam-compare)
        target_link_libraries(${target} PRIVATE wam_app_support wam::core)
        target_include_directories(${target} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/apps
            ${llama_SOURCE_DIR}/vendor/nlohmann)
        target_compile_options(${target} PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
        if(APPLE)
            set_target_properties(${target} PROPERTIES
                INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
        elseif(UNIX)
            set_target_properties(${target} PROPERTIES
                INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
        endif()
    endforeach()
    install(TARGETS wam-cli wam-bench wam-compare RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(WAM_BUILD_SERVER)
    install(PROGRAMS scripts/serving/wam_server.py
        DESTINATION ${CMAKE_INSTALL_BINDIR} RENAME wam-server)
    if(WAM_PROTO_DESCRIPTOR_AVAILABLE)
        install(PROGRAMS scripts/serving/wam_client.py
            DESTINATION ${CMAKE_INSTALL_BINDIR} RENAME wam-client)
    else()
        message(STATUS
            "wam: WAM_BUILD_SERVER=ON without protoc; installing wam-server only (wam-client unavailable)")
    endif()
endif()

if(WAM_BUILD_SERVER OR WAM_BUILD_GIGA_WORLD_POLICY)
    install(FILES scripts/serving/wam_rpc.py
        DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(WAM_PROTO_DESCRIPTOR_AVAILABLE AND
        (WAM_BUILD_SERVER OR WAM_BUILD_GIGA_WORLD_POLICY))
    install(FILES scripts/serving/wam_proto.py
        DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(WAM_BUILD_GIGA_WORLD_POLICY)
    if(WAM_PROTO_DESCRIPTOR_AVAILABLE)
        install(PROGRAMS
            integrations/giga-world-policy/run_inference_openloop_wam-cpp.sh
            DESTINATION ${CMAKE_INSTALL_BINDIR})
        install(PROGRAMS
            integrations/giga-world-policy/inference_openloop_wam_cpp.py
            DESTINATION
                ${CMAKE_INSTALL_DATADIR}/wam/integrations/giga-world-policy)
        install(FILES
            requirements/giga-world-policy.txt
            DESTINATION
                ${CMAKE_INSTALL_DATADIR}/wam/integrations/giga-world-policy)
        install(DIRECTORY integrations/giga-world-policy/upstream
            DESTINATION
                ${CMAKE_INSTALL_DATADIR}/wam/integrations/giga-world-policy)
        message(STATUS "wam: GigaWorld-Policy compatibility adapter enabled")
    else()
        message(STATUS
            "wam: GigaWorld-Policy adapter requested but protoc is unavailable; integration will not be installed")
    endif()
endif()

set(wam_export_targets wam_core wam_c_api ggml ggml-base)
get_target_property(wam_ggml_public_headers ggml PUBLIC_HEADER)
set(wam_ggml_public_headers_absolute)
foreach(header IN LISTS wam_ggml_public_headers)
    if(IS_ABSOLUTE "${header}")
        list(APPEND wam_ggml_public_headers_absolute "${header}")
    else()
        list(APPEND wam_ggml_public_headers_absolute "${llama_SOURCE_DIR}/ggml/${header}")
    endif()
endforeach()
set_target_properties(ggml PROPERTIES PUBLIC_HEADER "${wam_ggml_public_headers_absolute}")
if(TARGET ggml-cpu)
    list(APPEND wam_export_targets ggml-cpu)
endif()
if(TARGET ggml-cuda)
    list(APPEND wam_export_targets ggml-cuda)
endif()
if(APPLE)
    set(wam_install_rpath "@loader_path")
elseif(UNIX)
    set(wam_install_rpath "$ORIGIN")
endif()
if(DEFINED wam_install_rpath)
    foreach(target IN LISTS wam_export_targets)
        set_target_properties(${target} PROPERTIES INSTALL_RPATH "${wam_install_rpath}")
    endforeach()
endif()
install(TARGETS ${wam_export_targets}
    EXPORT wamTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY include/wam DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES proto/wam.proto DESTINATION ${CMAKE_INSTALL_DATADIR}/wam/proto)
if(WAM_PROTO_DESCRIPTOR_AVAILABLE)
    install(FILES "${WAM_PROTO_DESCRIPTOR}"
        DESTINATION ${CMAKE_INSTALL_DATADIR}/wam/proto)
endif()
install(FILES README.md LICENSE THIRD_PARTY_NOTICES.md
    DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/wam)
install(EXPORT wamTargets
    FILE wamTargets.cmake
    NAMESPACE wam::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/wam)

configure_package_config_file(
    cmake/WamConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/wamConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/wam)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/wamConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/wamConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/wamConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/wam)
