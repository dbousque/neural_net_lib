
#include "neural_network.h"
#include "mnist_reader.h"

void	malloc_error(void)
{
	printf("allocation failed.\n");
	exit(1);
}

double	sigmoid(double value)
{
	if (value > 20.0)
		return (1.0);
	else if (value < -20.0)
		return (0.0);
	return (1.0 / (1.0 + exp(-value)));
}

double	get_cost(double *out, double *exp_out, int length)
{
	double	res;
	int		i;

	res = 0.0;
	i = 0;
	while (i < length)
	{
		res += (exp_out[i] - out[i]) * (exp_out[i] - out[i]);
		i++;
	}
	return (res / length);
}

t_neural_net	*new_neural_net(int *nb_neurons_per_layer, int nb_layers, double (*activation_func)(double), 
																			double (*cost_func)(double *, double *, int))
{
	t_neural_net	*neural_net;
	int				i;
	int				x;
	int				y;

	if (!(neural_net = (t_neural_net*)malloc(sizeof(t_neural_net))))
		malloc_error();
	if (!(neural_net->layers = (t_layer**)malloc(sizeof(t_layer*) * nb_layers)))
		malloc_error();
	neural_net->nb_layers = nb_layers;
	i = 0;
	while (i < nb_layers)
	{
		if (!(neural_net->layers[i] = (t_layer*)malloc(sizeof(t_layer))))
			malloc_error();
		if (i > 0)
		{
			if (!(neural_net->layers[i]->neurons = (double**)malloc(sizeof(double*) * (nb_neurons_per_layer[i] + 1))))
				malloc_error();
			if (!(neural_net->layers[i]->neurons_out = (double*)malloc(sizeof(double) * (nb_neurons_per_layer[i] + 1))))
				malloc_error();
		}
		if (!(neural_net->layers[i]->neurons_out_act = (double*)malloc(sizeof(double) * (nb_neurons_per_layer[i] + 1))))
			malloc_error();
		if (!(neural_net->layers[i]->errors = (double*)malloc(sizeof(double) * (nb_neurons_per_layer[i]))))
				malloc_error();
		x = 0;
		while (i > 0 && x < nb_neurons_per_layer[i])
		{
			if (!(neural_net->layers[i]->neurons[x] = (double*)malloc(sizeof(double) * (nb_neurons_per_layer[i - 1] + 1))))
				malloc_error();
			y = 0;
			while (y < nb_neurons_per_layer[i - 1] + 1)
			{
				neural_net->layers[i]->neurons[x][y] = ((float)rand()/(float)(RAND_MAX)) * 2.0 - 1.0;
				y++;
			}
			x++;
		}
		neural_net->layers[i]->nb_neurons = nb_neurons_per_layer[i];
		neural_net->layers[i]->neurons_out_act[nb_neurons_per_layer[i]] = 1.0;
		i++;
	}
	neural_net->activation_func = activation_func;
	neural_net->cost_func = cost_func;
	neural_net->small_change = 0.01;
	return (neural_net);
}

/*double	dodo_ddot(int nb, double *arr1, int step1, double *arr2, int step2)
{
	double	res;

	(void)step2;
	(void)step1;
	res = 0.0;
	nb--;
	while (nb >= 0)
	{
		res += arr1[nb] * arr2[nb];
		nb--;
	}
	return (res);
}*/

double	*get_output(t_neural_net *neural_net, double *input)
{
	int		x;
	int		y;

	neural_net->layers[0]->neurons_out_act = input;
	x = 1;
	while (x < neural_net->nb_layers)
	{
		y = 0;
		while (y < neural_net->layers[x]->nb_neurons)
		{
			neural_net->layers[x]->neurons_out[y] = 1.0;
			neural_net->layers[x]->neurons_out[y] = cblas_ddot(neural_net->layers[x - 1]->nb_neurons + 1,
									neural_net->layers[x - 1]->neurons_out_act, 1, neural_net->layers[x]->neurons[y], 1);
			neural_net->layers[x]->neurons_out_act[y] = neural_net->activation_func(neural_net->layers[x]->neurons_out[y]);
			y++;
		}
		x++;
	}
	return (neural_net->layers[neural_net->nb_layers - 1]->neurons_out_act);
}

/*void	hadamard(double *a, double *b, double *res, int nb)
{
	nb--;
	while (nb >= 0)
	{
		res[nb] = a[nb] * b[nb];
		nb--;
	}
}*/

double	partial_der_sigmoid(t_neural_net *neural_net, int layer_nb, int neuron_nb)
{
	double	ori_val;
	double	res;

	ori_val = neural_net->layers[layer_nb]->neurons_out[neuron_nb];
	neural_net->layers[layer_nb]->neurons_out[neuron_nb] += neural_net->small_change;
	res = (neural_net->activation_func(neural_net->layers[layer_nb]->neurons_out[neuron_nb]) - neural_net->layers[layer_nb]->neurons_out_act[neuron_nb]) / neural_net->small_change;
	neural_net->layers[layer_nb]->neurons_out[neuron_nb] = ori_val;
	return (res);
}

void	compute_errors(t_neural_net *neural_net, double *exp_out)
{
	int		l;
	int		i;
	int		x;

	i = 0;
	while (i < neural_net->layers[neural_net->nb_layers - 1]->nb_neurons)
	{
		neural_net->layers[neural_net->nb_layers - 1]->errors[i] = (neural_net->layers[neural_net->nb_layers - 1]->neurons_out_act[i] - exp_out[i]) *
																											partial_der_sigmoid(neural_net, neural_net->nb_layers - 1, i);
		i++;
	}
	l = neural_net->nb_layers - 2;
	while (l > 0)
	{
		i = 0;
		while (i < neural_net->layers[l]->nb_neurons)
		{
			x = 0;
			neural_net->layers[l]->errors[i] = 0.0;
			while (x < neural_net->layers[l + 1]->nb_neurons)
			{
				neural_net->layers[l]->errors[i] += neural_net->layers[l + 1]->neurons[x][i] * neural_net->layers[l + 1]->errors[x];
				x++;
			}
			neural_net->layers[l]->errors[i] *= partial_der_sigmoid(neural_net, l, i);
			i++;
		}
		l--;
	}
}

void	update_gradient_with_input(t_neural_net *neural_net, double ***gradient, double *input, double *exp_out)
{
	int		x;
	int		y;
	int		z;

	get_output(neural_net, input);
	compute_errors(neural_net, exp_out);
	x = 1;
	while (x < neural_net->nb_layers)
	{
		y = 0;
		while (y < neural_net->layers[x]->nb_neurons)
		{
			z = 0;
			while (z < neural_net->layers[x - 1]->nb_neurons)
			{
				gradient[x][y][z] += neural_net->layers[x - 1]->neurons_out_act[z] * neural_net->layers[x]->errors[y];
				z++;
			}
			gradient[x][y][z] += neural_net->layers[x]->errors[y];
			y++;
		}
		x++;
	}
}

void	get_gradient(t_neural_net *neural_net, double ***gradient, double **inputs, double **exp_outs, int mini_batch_size, int start)
{
	int		x;
	int		y;
	int		z;
	int		i;

	x = 1;
	while (x < neural_net->nb_layers)
	{
		y = 0;
		while (y < neural_net->layers[x]->nb_neurons)
		{
			z = 0;
			while (z < neural_net->layers[x - 1]->nb_neurons + 1)
			{
				gradient[x][y][z] = 0.0;
				z++;
			}
			y++;
		}
		x++;
	}
	i = start;
	while (i < start + mini_batch_size)
	{
		update_gradient_with_input(neural_net, gradient, inputs[i], exp_outs[i]);
		i++;
	}
}

double	***new_gradient(t_neural_net *neural_net)
{
	double	***gradient;
	int		x;
	int		y;

	if (!(gradient = (double***)malloc(sizeof(double**) * neural_net->nb_layers)))
		malloc_error();
	x = 1;
	while (x < neural_net->nb_layers)
	{
		if (!(gradient[x] = (double**)malloc(sizeof(double*) * neural_net->layers[x]->nb_neurons)))
			malloc_error();
		y = 0;
		while (y < neural_net->layers[x]->nb_neurons)
		{
			if (!(gradient[x][y] = (double*)malloc(sizeof(double) * (neural_net->layers[x - 1]->nb_neurons + 1))))
				malloc_error();
			y++;
		}
		x++;
	}
	return (gradient);
}

void	update_weights(t_neural_net *neural_net, double ***gradient, int mini_batch_size, double learning_rate)
{
	int		x;
	int		y;
	int		z;

	x = 1;
	while (x < neural_net->nb_layers)
	{
		y = 0;
		while (y < neural_net->layers[x]->nb_neurons)
		{
			z = 0;
			while (z < neural_net->layers[x - 1]->nb_neurons + 1)
			{
				neural_net->layers[x]->neurons[y][z] += -(gradient[x][y][z] / mini_batch_size) * learning_rate;
				z++;
			}
			y++;
		}
		x++;
	}
}

double	*predict(t_neural_net *neural_net, double *input)
{
	return (get_output(neural_net, input));
}

double	score(t_neural_net *neural_net, double **inputs, double **expected_outputs, int nb_inputs)
{
	int		i;
	int		x;
	double	*out;
	int		ind_max1;
	int		ind_max2;
	int		correct;
	int		incorrect;

	correct = 0;
	incorrect = 0;
	i = 0;
	while (i < nb_inputs)
	{
		out = predict(neural_net, inputs[i]);
		ind_max1 = -1;
		ind_max2 = -1;
		x = 0;
		while (x < neural_net->layers[neural_net->nb_layers - 1]->nb_neurons)
		{
			if (ind_max1 == -1 || out[x] > out[ind_max1])
				ind_max1 = x;
			x++;
		}
		x = 0;
		while (x < neural_net->layers[neural_net->nb_layers - 1]->nb_neurons)
		{
			if (ind_max2 == -1 || expected_outputs[i][x] > expected_outputs[i][ind_max2])
				ind_max2 = x;
			x++;
		}
		if (ind_max1 == ind_max2)
			correct++;
		else
			incorrect++;
		i++;
	}
	return ((float)incorrect / (correct + incorrect) * 100.0);
}

void	gradient_descent(t_neural_net *neural_net, double **inputs, double **exp_outs, int nb_inputs, int mini_batch_size, int nb_epochs, double learning_rate)
{
	double	***gradient;
	int		i;
	int		x;

	gradient = new_gradient(neural_net);
	i = 0;
	while (i < nb_epochs)
	{
		x = 0;
		while (x < nb_inputs)
		{
			get_gradient(neural_net, gradient, inputs, exp_outs, mini_batch_size, x);
			update_weights(neural_net, gradient, mini_batch_size, learning_rate);
			x += mini_batch_size;
		}
		i++;
		printf("EPOCH %d ; error rate : %.2f%%\n", i, score(neural_net, inputs, exp_outs, nb_inputs));
	}
}

void	create_train_save_nn(int *nb_neurons_per_layer, int nb_layers, double **inputs, double **exp_outs,
			int nb_inp, int distortions, int nb_epochs, double learning_rate, int mini_batch_size, char *filename)
{
	t_neural_net	*neural_net;

	if (distortions)
	{
		generate_distortions(&inputs, &exp_outs, nb_inp);
		nb_inp *= 5;
	}
	neural_net = new_neural_net(nb_neurons_per_layer, nb_layers, sigmoid, get_cost);
	gradient_descent(neural_net, inputs, exp_outs, nb_inp, mini_batch_size, nb_epochs, learning_rate);
	save_neural_network_to_file(neural_net, filename);
}

/* ________________________________ END OF NN LIBRARY _______________________________*/

#define LEARNING_RATE 3.0
#define LAST_LAYER_NB_NEURONS 10
#define FISRT_LAYER_NB_NEURONS 784
#define NB_EPOCHS 60
#define MINI_BATCH_SIZE 10
#define COST_FUNC get_cost
#define ACTIVATION_FUNC sigmoid
#define NB_INPUTS 10000
#define NB_INPUTS2 50000
#define DISTORTIONS 1

int		main(void)
{
	t_image			**images;
	double			**inputs;
	double			**expected_outputs;
	double			**inputs2;
	double			**expected_outputs2;
	t_neural_net	*neural_net;
	int				nb_neurons_per_layer[] = {FISRT_LAYER_NB_NEURONS, 100, 20, LAST_LAYER_NB_NEURONS};
	int				nb_inputs;

	//srand(time(NULL));
	images = get_images();
	inputs = get_inputs_from_images(images, NB_INPUTS);
	expected_outputs = get_outputs_from_images(images, NB_INPUTS, LAST_LAYER_NB_NEURONS);
	nb_inputs = NB_INPUTS;
	if (DISTORTIONS)
	{
		generate_distortions(&inputs, &expected_outputs, NB_INPUTS);
		//write_image_to_file(inputs[0], "ori.tif");
		//write_image_to_file(inputs[nb_inputs], "right.tif");
		//write_image_to_file(inputs[nb_inputs * 2], "left.tif");
		//write_image_to_file(inputs[nb_inputs * 3], "down.tif");
		//write_image_to_file(inputs[nb_inputs * 4], "up.tif");
		nb_inputs *= 5;
	}
	inputs2 = get_inputs_from_images(images, NB_INPUTS2);
	expected_outputs2 = get_outputs_from_images(images, NB_INPUTS2, LAST_LAYER_NB_NEURONS);
	neural_net = new_neural_net(nb_neurons_per_layer, 4, ACTIVATION_FUNC, COST_FUNC);
	//print_neural_network(neural_net);
	printf("ori_score : %.2f%%\n", score(neural_net, inputs, expected_outputs, nb_inputs));
	gradient_descent(neural_net, inputs, expected_outputs, nb_inputs, MINI_BATCH_SIZE, NB_EPOCHS, LEARNING_RATE);
	printf("score on 10000 : %.2f%%\n", score(neural_net, inputs, expected_outputs, nb_inputs));
	printf("score on 60000 : %.2f%%\n", score(neural_net, inputs2, expected_outputs2, NB_INPUTS2));
	//print_neural_network(neural_net);
	save_neural_network_to_file(neural_net, "nn4.nn");
	return (0);
}