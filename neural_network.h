

#ifndef NEURAL_NETWORK_H
# define NEURAL_NETWORK_H

# include <stdlib.h>
# include <stdio.h>
# include <cblas.h>
# include <math.h>
# include <string.h>
# include <time.h>
# include <fcntl.h>
# include <unistd.h>

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

//double			**get_inputs_from_images(t_image **images, int nb);
//double			**get_outputs_from_images(t_image **images, int nb, int nb2);
void			generate_distortions(double ***inputs, double ***outputs, int nb, int row_size, int column_size);
void			save_neural_network_to_file(t_neural_net *neural_net, char *filename);
//void			write_image_to_file(double *image, char *filename, int size);
void			create_train_save_nn(int *nb_neurons_per_layer, int nb_layers, double **inputs, double **exp_outs,
					int nb_inp, int distortions, int nb_epochs, double learning_rate, int mini_batch_size, char *filename);
void			gradient_descent(t_neural_net *neural_net, double **inputs, double **exp_outs,
						int nb_inputs, int mini_batch_size, int nb_epochs, double learning_rate);
t_neural_net	*new_neural_net(int *nb_neurons_per_layer, int nb_layers, double (*activation_func)(double), 
																			double (*cost_func)(double *, double *, int));
double			sigmoid(double value);
double			get_cost(double *out, double *exp_out, int length);
double			score(t_neural_net *neural_net, double **inputs, double **expected_outputs, int nb_inputs);
void			save_neural_network_to_file(t_neural_net *neural_net, char *filename);
double			*predict(t_neural_net *neural_net, double *input);

#endif