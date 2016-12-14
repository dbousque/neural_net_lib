

#ifndef NEURAL_NETWORK_H
# define NEURAL_NETWORK_H

# include <stdlib.h>
# include <stdio.h>
# include <cblas.h>
# include <math.h>

typedef struct	s_layer
{
	double		**neurons;
	double		*neurons_out;
	double		*neurons_out_act;
	int			nb_neurons;
}				t_layer;

typedef struct	s_neural_net
{
	int			nb_layers;
	t_layer		**layers;
	double		(*activation_func)(double);
	double		(*cost_func)(double *, double *, int);
	double		small_change;
}				t_neural_net;

#endif