# Computer Generated Holograms optimized using Automatic Differentiation (CGH-autodiff)
This repository is a collection of multiple projects that utilize automatic differentiation to optimize Computer Generated Holograms.  
There are two projects included (2021/07/23).

## Multiple Binary Optimization

### [Project Page](https://digitalnature.slis.tsukuba.ac.jp/2021/07/multiple-binary-hologram-optimization/) | [Paper]() | [Slide](https://speakerdeck.com/digitalnaturegroup/gradient-based-optimization-of-time-multiplexed-binary-computer-generated-holograms-by-digital-mirror-device-digital-holography-and-three-dimensional-imaging-at-osa-imaging-and-applied-optics-congress-oral-presentation-by-kenta-yamamoto)
<b >"Gradient-based Optimization of Time-Multiplexed Binary Computer-Generated Holograms by Digital Mirror Device"</b>  
[Kenta Yamamoto](https://digitalnature.slis.tsukuba.ac.jp/2017/04/kenta-yamamoto/), [Yoichi Ochiai](https://digitalnature.slis.tsukuba.ac.jp/2018/09/yoichi-ochiai/)

This project was orally presented on [Digital Holography and Three-Dimensional Imaging](https://www.osa.org/en-us/meetings/osa_meetings/osa_imaging_and_applied_optics_congress/program/digital_holography_and_three-dimensional_imaging/) at [OSA Imaging and Applied Optics Congress](https://www.osa.org/en-us/meetings/osa_meetings/osa_imaging_and_applied_optics_congress/).

In this research, we utilize the binary amplitude spatial light modulator (SLM), which has a high refresh rate such as Digital Mirror Device (DMD), to optimize the hologram in time-multiplexing to display a high-definition reproduced image.  
The propagation results of multiple binary amplitude holograms are added, and the difference (loss) between the summed image and the target image is taken to optimize each binary amplitude hologram.

<br>

## Neural Holgoraphy with TensorFlow
This project was developed based on the method of the paper <b>"Neural Holography with Camera-in-the-loop Training"</b> ([Paper Link](http://www.computationalimaging.org/wp-content/uploads/2020/08/NeuralHolography_SIGAsia2020.pdf), [Source Code](https://github.com/computational-imaging/neural-holography), [Project Page](http://www.computationalimaging.org/publications/neuralholography/)).  
PyTorch was used in "Neural Holography"; however, we implemented it in Tensorflow to use a 2D Fourier Transform that supports automatic differentiation.


