

#ifndef NEURAL_NET_WRAPPER_H
# define NEURAL_NET_WRAPPER_H

# include <Python.h>
# include "neural_network.h"

static char module_docstring[] =
    "Neural network simple library.";
static char new_neural_net_docstring[] =
    "Calculate the chi-squared of some data given a model.";

static PyObject	*neural_net_new_neural_net(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"new_neural_net", neural_net_new_neural_net, METH_VARARGS, new_neural_net_docstring},
    {NULL, NULL, 0, NULL}
};

#endif