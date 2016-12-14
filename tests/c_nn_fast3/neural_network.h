

#ifndef NEURAL_NETWORK_H
# define NEURAL_NETWORK_H

# include <stdlib.h>
# include <stdio.h>
# include <cblas.h>
# include <math.h>
# include <string.h>
# include "mnist_reader.h"

typedef struct	s_layer
{
	double		**neurons;
	double		*neurons_out;
	double		*neurons_out_act;
	double		*errors;
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

double			**get_inputs_from_images(t_image **images, int nb);
double			**get_outputs_from_images(t_image **images, int nb, int nb2);
double			*decal_one_right(double *inp, int inp_size);
double			*decal_one_left(double *inp, int inp_size);
double			*decal_one_down(double *inp, int row_size);
double			*decal_one_up(double *inp, int row_size);
void			generate_distortions(double ***inputs, double ***outputs, int nb);
void			write_image_to_file(double *image, char *filename);
void			save_neural_network_to_file(t_neural_net *neural_net, char *filename);
void			write_image_to_file(double *image, char *filename);
void			create_train_save_nn(int *nb_neurons_per_layer, int nb_layers, double **inputs, double **exp_outs,
					int nb_inp, int distortions, int nb_epochs, double learning_rate, int mini_batch_size, char *filename);

#endif