project('vsdlisr', 'cpp',
  default_options: ['buildtype=release', 'warning_level=2', 'b_lto=true', 'b_ndebug=if-release', 'cpp_std=c++17'],
  meson_version: '>=0.51.0',
  version: '1'
)

cxx = meson.get_compiler('cpp')

gcc_syntax = cxx.get_argument_syntax() == 'gcc'

if get_option('buildtype') == 'release'
  add_project_arguments(gcc_syntax ? ['-fno-math-errno', '-fno-trapping-math'] : '/GS-', language: 'cpp')
endif

lib_dirs = []

if gcc_syntax
  vapoursynth_dep = dependency('vapoursynth', version: '>=55').partial_dependency(compile_args: true, includes: true)
  install_dir = vapoursynth_dep.get_variable(pkgconfig: 'libdir') / 'vapoursynth'
else
  vapoursynth_dep = []
  install_dir = get_option('libdir') / 'vapoursynth'
  lib_dirs += include_directories('C:/Program Files/VapourSynth/sdk/include/')
endif

cuda_dep = cxx.find_library('cudart',
  dirs: 'C:/ProgramData/NVIDIA Corporation/NVIDIA NGX SDK/1.1/Samples/Extern/cuda/lib/x64/',
  header_include_directories: include_directories('C:/ProgramData/NVIDIA Corporation/NVIDIA NGX SDK/1.1/Samples/Extern/cuda/include/'),
  has_headers: ['cuda.h', 'cuda_runtime.h']
)
ngx_dep = cxx.find_library('nvsdk_ngx_d',
  dirs: 'C:/ProgramData/NVIDIA Corporation/NVIDIA NGX SDK/1.1/lib/x64/',
  header_include_directories: include_directories('C:/ProgramData/NVIDIA Corporation/NVIDIA NGX SDK/1.1/Include'),
  has_headers: ['nvsdk_ngx.h', 'nvsdk_ngx_defs.h']
)

lib_dirs += include_directories('C:/ProgramData/NVIDIA Corporation/NVIDIA NGX SDK/1.1/Include')
lib_dirs += include_directories('C:/ProgramData/NVIDIA Corporation/NVIDIA NGX SDK/1.1/Samples/Extern/cuda/include/')

sources = [
  'vsdlisr/DLISR.cpp',
  'vsdlisr/DLISR.h',
]

libs = []

shared_module('vsdlisr', sources,
  dependencies: [vapoursynth_dep, cuda_dep, ngx_dep],
  include_directories: lib_dirs,
  link_with: libs,
  install: true,
  install_dir: install_dir,
  gnu_symbol_visibility: 'hidden'
)
