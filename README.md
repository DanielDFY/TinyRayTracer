# TinyRayTracer
 A tiny ray tracer with CUDA.



To produce image as `.png` format, [stb](https://github.com/nothings/stb) is included as a submodule under `include/stb`.



When the CUDA code runs for longer than a few seconds, you may notice errors like unspecified launch failure or `cudaErrorLaunchTimeout`, try to set `WDDM TDR` in Nsight Monitor from `true` to `false` as admin.