# dlisr

dlisr using Nvidia ngx

Just a test to get it working on rtx 4000 series

notes:

- only supported formats is rgb ui8, 16f, 32f
- 32f is broken, didn't try 16f
- the working nvngx_dlisr.dll for rtx 4000 can be found from the latest driver? or cuda package? (not ngx sdk)
- this version of [ngx](https://developer.nvidia.com/rtx/ngx/download) (1.1 latest) is only available for windows

## Usage

    vsdlisr.DLISR(vnode clip[, int rfactor=2])

- clip: Clip to process. Must be 8 bit RGB.

- rfactor: powers of 2

## Compilation

```
meson build
ninja -C build
ninja -C build install
```
