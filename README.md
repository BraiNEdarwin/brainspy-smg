# brainspy-smg [![CircleCI](https://dl.circleci.com/status-badge/img/gh/BraiNEdarwin/brainspy-smg/tree/master.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/BraiNEdarwin/brainspy-smg/tree/master) [![Tools](https://img.shields.io/badge/brains-py-darkblue.svg)](https://github.com/BraiNEdarwin/brains-py) [![Theory](https://img.shields.io/badge/brainspy-tasks-lightblue.svg)](https://github.com/BraiNEdarwin/brainspy-tasks)

Automatised generation of deep neural-network based surrogate models of multi-terminal nano-electronic devices, particularly it is focused on dopant-network processing units (DNPUs) [1]. The library provides support for the whole life-cycle of a surrogate model:

- **Device preparation:** Depending on its properties, such as the material type, devices can exhibit different behaviours. Before the training step, there is a device preparation step, where the IV curves for different selected activation electrodes are studied in different conditions. This library helps visualising the IV curves not only for a single device, but for multiple devices at the same time, enabling to study them simultaneously. 
- **Data acquisition:** The data are sampled using sinusoidal or triangular modulated input signals to minimize capacitive transient effects. The frequencies, amplitudes and offsets of these functions for each activation electrode are given by the voltage ranges chosen during the IV curve observation stage. Frequencies are proportional to the  square root of prime numbers. Since the ratio of any two frequencies is irrational, it guarantees a good coverage of the multi-dimensional input voltage space and prevents any recurrence of voltage combinations. In order to have an even quicker coverage, the phase of inputs signals can be randomly shifted. More information can be found in [3].
- **Model training:** A model is trained using a Pytorch based neural network. The user can customise the size of the neural network as well as the activation functions used for this model. The tool produces error plots of the trained surrogate model on all training, validation and test datasets. It also saves relevant information about how the data was acquired, how the model was trained, and other relevant electrode information that is required for using surrogate models in brains-py.
- **Model maintenance:** It could be that devices change its behaviour after receiving abrupt changes in temperature or high voltages. In order to check if a model is still behaving as the original hardware, some consistency checks are provided, that help understand the differences between the actual signal of the device, and the signal from the surrogate model, compared to the original data that was gathered for training the device.





## 1. Instructions

You can find detailed instructions for the following topics on the [wiki](https://github.com/BraiNEdarwin/brainspy-smg/wiki):

- [Introduction](https://github.com/BraiNEdarwin/brainspy-smg/wiki/A.-Introduction): Provides a general description behind the background of this project project. These instructions are strongly recommended for new students joining the research group.
- [Package description](https://github.com/BraiNEdarwin/brainspy-smg/wiki/B.-Package-description): Gives more information on how the package is structured and the particular usage of files.
- [Installation instructions](https://github.com/BraiNEdarwin/brainspy-smg/wiki/C.-Installation-Instructions): How to correctly install this package
- [User instructions and usage examples](https://github.com/BraiNEdarwin/brainspy-smg/wiki/D.-User-Instructions-and-usage-examples): Instructions for users and examples on how brains-py can be used for different purposes
- [Developer instructions](https://github.com/BraiNEdarwin/brainspy-smg/wiki/E.-Developer-Instructions): Instructions for people wanting to develop brains-py

## 2. License and libraries

This code is released under the GNU GENERAL PUBLIC LICENSE Version 3. Click [here](https://github.com/BraiNEdarwin/brains-py/blob/master/doc/LICENSE) to see the full license.
The package relies on the following libraries:

- General support libraries:
  - PyTorch, Pytorch-Lightning, NumPy, tensorboard, more_itertools  and matplotlib
- It also uses brains-py and all its related libraries.

## 3. Related scientific publications

[1] Chen, T., van Gelder, J., van de Ven, B., Amitonov, S. V., de Wilde, B., Euler, H. C. R., ... & van der Wiel, W. G. (2020). Classification with a disordered dopant-atom network in silicon. *Nature*, *577*(7790), 341-345. [Classification with a disordered dopant-atom network in silicon | Nature](https://doi.org/10.1038/s41586-019-1901-0)

[2] HCR Euler, U Alegre-Ibarra, B van de Ven, H Broersma, PA Bobbert and WG van der Wiel (2020). Dopant Network Processing Units: Towards Efficient Neural-network Emulators with High-capacity Nanoelectronic Nodes. [https://arxiv.org/abs/2007.12371](https://arxiv.org/abs/2007.12371)](https://arxiv.org/abs/2007.12371%5D(https://arxiv.org/abs/2007.12371))

[3] HCR Euler, MN Boon, JT Wildeboer, B van de Ven, T Chen, H Broersma, PA Bobbert, WG van der Wiel (2020). A Deep-Learning Approach to Realising Functionality in Nanoelectronic Devices. [A deep-learning approach to realizing functionality in nanoelectronic devices | Nature Nanotechnology](https://doi.org/10.1038/s41565-020-00779-y)

## 4. Acknowledgements

This package has been created and it is maintained by the [Brains](https://www.utwente.nl/en/brains/) team of the [NanoElectronics](https://www.utwente.nl/en/eemcs/ne/) research group at the University of Twente. It has been designed by:

- **Dr. Unai Alegre-Ibarra**, [@ualegre](https://github.com/ualegre) ([u.alegre@utwente.nl](mailto:u.alegre@utwente.nl)): Project lead, including requirements, design, implementation, maintenance, linting tools, testing and documentation (Jupyter notebooks, Wiki and supervision of file by file documentation).
- **Dr. Hans Christian Ruiz-Euler**, [@hcruiz](https://github.com/hcruiz) ([h.ruiz@utwente.nl](mailto:h.ruiz@utwente.nl)): Initial design and implementation of major features both in this repository and in the legacy [SkyNEt](https://github.com/BraiNEdarwin/SkyNEt) repository and in this one.

With the contribution of:

- **Marcus Boon**: [@Mark-Boon](https://github.com/Mark-Boon):  The initial legacy code for some of the process, as found in [SkyNEt](https://github.com/BraiNEdarwin/SkyNEt) repository.
- **Srikumar Sastry**, [Vishu26](https://github.com/Vishu26) ([s.s.sastry@student.utwente.nl](mailto:s.s.sastry@student.utwente.nl)) : Testing and identification of bugs. 
- **Dr. ir. Michel P. de Jong** [@xX-Michel-Xx](https://github.com/xX-Michel-Xx) ([m.p.dejong@utwente.nl](mailto:m.p.dejong@utwente.nl)): Testing and identification of bugs, especially on the installation procedure.
- **Mohamadreza Zolfagharinejad** [@mamrez](https://github.com/mamrez) ([m.zolfagharinejad@utwente.nl](mailto:m.zolfagharinejad@utwente.nl)): Writing of some of the examples in Jupyter notebooks (IV curves and surrogate model generation).
- **Antonio J. Sousa de Almeida** [@ajsousal](https://github.com/ajsousal) ([a.j.sousadealmeida@utwente.nl](mailto:a.j.sousadealmeida@utwente.nl)):: Checking and upgrading drivers and National Instruments equipment from the labs.
- **Bram van de Ven**, [@bbroo1](https://github.com/bbroo1) ([b.vandeven@utwente.nl](mailto:b.vandeven@utwente.nl)) : General improvements and testing of the different hardware drivers and devices and documentation.

Other minor contributions might have been added, in form of previous scripts that have been improved and restructured from [SkyNEt](https://github.com/BraiNEdarwin/SkyNEt), and the authorship remains of those people who collaborated in it.

This project has received financial support from:

- **University of Twente**
- **Dutch Research Council**
  - HTSM grant no. 16237
  - Natuurkunde Projectruimte grant no. 680-91-114
- **Toyota Motor Europe N.V.**
