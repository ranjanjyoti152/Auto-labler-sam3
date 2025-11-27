# SAM3 Patch

This directory vendors the `sam3.sam` subpackage from [facebookresearch/sam3](https://github.com/facebookresearch/sam3) to work around the current wheel published on GitHub/PyPI, which omits the `sam3.sam` sources required at runtime. The files are kept unmodified under the original MIT license (see `LICENSE`).

Whenever you update the upstream dependency, re-copy the contents of `sam3/sam` to ensure the detector keeps matching the released model.
