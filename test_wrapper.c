

#include "test_wrapper.h"

PyMODINIT_FUNC init_test(void)
{
    PyObject *m = Py_InitModule3("_test", module_methods, module_docstring);
    if (m == NULL)
        return;
}

static PyObject	*test_test_func(PyObject *self, PyObject *args)
{
	int			val1;
	int			val2;
	PyObject	*res;

    PyArg_ParseTuple(args, "ii", &val1, &val2);
    res = Py_BuildValue("i", test_func(val1, val2));
    return (res);
}