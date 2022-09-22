"""
Automatised generation of deep neural-network based surrogate models of multi-terminal nano-electronic devices, particularly it is focused on dopant-network processing units (DNPUs) [1]. The library provides support for the whole life-cycle of a surrogate model:

-Device preparation: Depending on its properties, such as the material type, devices can exhibit different behaviours. Before the training step, there is a device preparation step, where the IV curves for different selected activation electrodes are studied in different conditions. This library helps visualising the IV curves not only for a single device, but for multiple devices at the same time, enabling to study them simultaneously.

-Data acquisition: The data are sampled using sinusoidal or triangular modulated input signals to minimize capacitive transient effects. The frequencies, amplitudes and offsets of these functions for each activation electrode are given by the voltage ranges chosen during the IV curve observation stage. Frequencies are proportional to the square root of prime numbers. Since the ratio of any two frequencies is irrational, it guarantees a good coverage of the multi-dimensional input voltage space and prevents any recurrence of voltage combinations. In order to have an even quicker coverage, the phase of inputs signals can be randomly shifted. More information can be found in [3].

-Model training: A model is trained using a Pytorch based neural network. The user can customise the size of the neural network as well as the activation functions used for this model. The tool produces error plots of the trained surrogate model on all training, validation and test datasets. It also saves relevant information about how the data was acquired, how the model was trained, and other relevant electrode information that is required for using surrogate models in brains-py.

-Model maintenance: It could be that devices change its behaviour after receiving abrupt changes in temperature or high voltages. In order to check if a model is still behaving as the original hardware, some consistency checks are provided, that help understand the differences between the actual signal of the device, and the signal from the surrogate model, compared to the original data that was gathered for training the device.

More information at: https://github.com/BraiNEdarwin/brains-py/wiki
"""
TEST_MODE = "SIMULATION_PC"
