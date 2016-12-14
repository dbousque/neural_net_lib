

#ifndef NEURAL_NETWORK_H
# define NEURAL_NETWORK_H

# include <math.h>
# include <stdlib.h>
# include <stdio.h>
# include <string.h>
# include <time.h>
# include <cblas.h>

typedef struct	s_neuron
{
	double		*weights;
	int			nb_weights;
}				t_neuron;

typedef struct	s_layer
{
	int			nb_neurons;
	t_neuron	**neurons;
	double		*neurons_values;
}				t_layer;

typedef struct	s_neural_net
{
	int			nb_layers;
	t_layer		**layers;
	double		learning_rate;
	double		small_value;
}				t_neural_net;

void			calculate_neuron_value(t_layer *layer, int neuron_nb, t_layer *previous_layer, double (*activation_func)(double));
void			set_first_layer(t_neural_net *neural_net, double *input);
void			get_gradient(t_neural_net *neural_net, double (*cost_func)(double[], double[], int),
										double (*activation_func)(double), double *expected_output,
															double *ret, double ***gradient);
void			update_weights(t_neural_net *neural_net, double ***gradient);
void			score(t_neural_net *neural_net, double **inputs, double **expected_outputs, int nb_inputs,
																double (*activation_func)(double));

#endif