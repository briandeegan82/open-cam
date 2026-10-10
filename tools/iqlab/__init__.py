"""Image-quality lab metrics for open-cam (SNR / dynamic range, IEEE P2020, CPIQ-style).

Every function works on plain NumPy arrays so it can score pbrt -> sensor -> ISP outputs and
real captures alike. Modules: :mod:`iqlab.snr`, :mod:`iqlab.p2020`, :mod:`iqlab.cpiq`,
:mod:`iqlab.dead_leaves`, :mod:`iqlab.geometry`. See docs/IQ_LAB.md.
"""
