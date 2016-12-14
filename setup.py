from distutils.core import setup, Extension

# compilation : python setup.py build_ext --inplace

setup(
	ext_modules=[Extension("neural_net", ["neural_network.c", "neural_net_wrapper.c"], extra_link_args=['-lblas', '-lm', '-O3'])]
)