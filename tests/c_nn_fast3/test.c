

#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>

int		main(void)
{
	double	*errors;
	double	a[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
	double	b[] = {2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0};
	int		i;

	errors = (double*)malloc(sizeof(double) * 10);
	cblas_dgemv(10, a, b, errors);
	i = 0;
	while (i < 10)
	{
		printf("%f\n", errors[i]);
		i++;
	}
	return (0);
}