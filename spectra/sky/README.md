# Sky model data

`hosek_wilkie_2012_spectral.npz` — the spectral coefficient tables of the Hosek–Wilkie
clear-sky model (`datasets`, shape (11 wavelengths, 2 albedos, 10 turbidities, 6 elevation
control points, 9 parameters), and `datasets_rad`, shape (11, 2, 10, 6)) at 320–720 nm in
40 nm steps, extracted verbatim from `ArHosekSkyModelData_Spectral.h` of the reference
implementation (shipped in `third_party/pbrt-v4/src/ext/skymodel/`) so the unit tests and the
scene builder do not need the pbrt submodule. Used by `tools/highway_spectral_sky.py`.

L. Hosek and A. Wilkie, "An Analytic Model for Full Spectral Sky-Dome Radiance",
ACM Trans. Graph. 31(4):95, 2012.

```
Copyright (c) 2012 - 2013, Lukas Hosek and Alexander Wilkie
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * None of the names of the contributors may be used to endorse or promote
      products derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
