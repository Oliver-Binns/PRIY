# Parallel Programming Tools for Exploring Immune System Development
## Final Year Project / Dissertation

The project aims to reimplement the [Peyer's Patch simulation](https://www.york.ac.uk/computational-immunology/software/ppsim/) using [FlameGPU](https://www.github.com/FlameGPU/FlameGPU.git).
The main aim behind this project is to produce a significant cut in the run-time of the original simulation by utilising GPGPU programming. From this, a mapping from the original platform model to the final FlameGPU model will hopefully allow for other biological simulations to more easily be implemented by [YCIL](http://ycil.org.uk/).

FlameGPU Dependencies
* CUDA SDK
* libxml2-utils
* xsltproc
