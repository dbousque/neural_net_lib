

#include "neural_net_wrapper.h"

#include <stdio.h>

PyMODINIT_FUNC initneural_net(void)
{
    PyObject *m = Py_InitModule3("neural_net", module_methods, module_docstring);
    if (m == NULL)
        return;
}

static PyObject	*neural_net_new_neural_net(PyObject *self, PyObject *args)
{
	PyObject	*res;
	PyObject	*nb_neurons_per_layer;
	//int			*nb_neurons;
	int			nb_layers;
	//PyObject	*iter;

	nb_neurons_per_layer = malloc(sizeof(PyObject));
	//printf("avant avant\n");
    //fflush(stdout);
    PyArg_ParseTuple(args, "O!", &PyList_Type, &nb_neurons_per_layer);
    nb_layers = PyList_Size(nb_neurons_per_layer);
    /*printf("apres\n");
    fflush(stdout);
	iter = PyObject_GetIter(nb_neurons_per_layer);
	printf("apres apres\n");
    fflush(stdout);
	while (1)
	{
		printf("boucle\n");
    	fflush(stdout);
		PyObject *next = PyIter_Next(iter);
		//printf("lol\n");
		if (!next)
			break;
		nb_layers++;
	}*/
	res = Py_BuildValue("i", nb_layers);
    return (res);
}