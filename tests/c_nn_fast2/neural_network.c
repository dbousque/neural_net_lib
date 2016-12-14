
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

double	*get_output(t_neural_net *neural_net, double *input)
{
	int		x;
	int		y;

	static int i = 0;
	i++;

	neural_net->layers[0]->neurons_out_act = input;
	//printf("CALLED\n");
	x = 1;
	while (x < neural_net->nb_layers)
	{
		y = 0;
		while (y < neural_net->layers[x]->nb_neurons)
		{
			neural_net->layers[x]->neurons_out[y] = cblas_ddot(neural_net->layers[x - 1]->nb_neurons + 1,
									neural_net->layers[x - 1]->neurons_out_act, 1, neural_net->layers[x]->neurons[y], 1);
			if (x == 2)
				//printf("nb_neurons : %d, y : %d\n", neural_net->layers[x]->nb_neurons, y);
			if (x == 2)
			{
				//printf("dot_product : %f\n", neural_net->layers[x]->neurons_out[y]);
			}
			neural_net->layers[x]->neurons_out_act[y] = neural_net->activation_func(neural_net->layers[x]->neurons_out[y]);
			y++;
		}
		x++;
	}
	//if (i == 5)
		//exit(0);
	//printf("\n");
	return (neural_net->layers[neural_net->nb_layers - 1]->neurons_out_act);
}

void	update_layer(t_neural_net *neural_net, int layer_nb)
{
	(void)layer_nb;
	(void)neural_net;
}

double	get_partial_derivative(t_neural_net *neural_net, int lay_nb, int neur_nb, int weight_nb, double ori_cost, double *exp_out)
{
	int		i;
	double	ori_out_act;
	double	ori_val;
	double	ori_out;
	double	partial_der;

	ori_val = neural_net->layers[lay_nb]->neurons[neur_nb][weight_nb];
	ori_out_act = neural_net->layers[lay_nb]->neurons_out_act[neur_nb];
	ori_out = neural_net->layers[lay_nb]->neurons_out[neur_nb];

	neural_net->layers[lay_nb]->neurons_out[neur_nb] -= neural_net->layers[lay_nb - 1]->neurons_out_act[weight_nb] * neural_net->layers[lay_nb]->neurons[neur_nb][weight_nb];
	neural_net->layers[lay_nb]->neurons[neur_nb][weight_nb] += neural_net->small_change;

	neural_net->layers[lay_nb]->neurons_out[neur_nb] += neural_net->layers[lay_nb - 1]->neurons_out_act[weight_nb] * neural_net->layers[lay_nb]->neurons[neur_nb][weight_nb];
	neural_net->layers[lay_nb]->neurons_out_act[neur_nb] = neural_net->activation_func(neural_net->layers[lay_nb]->neurons_out[neur_nb]);
	i = 0;
	if (lay_nb < neural_net->nb_layers - 1)
	{
		while (i < neural_net->layers[lay_nb + 1]->nb_neurons)
		{
			neural_net->layers[lay_nb + 1]->neurons_out[i] -= ori_out_act * neural_net->layers[lay_nb + 1]->neurons[i][neur_nb];
			neural_net->layers[lay_nb + 1]->neurons_out[i] += neural_net->layers[lay_nb]->neurons_out_act[neur_nb] * neural_net->layers[lay_nb + 1]->neurons[i][neur_nb];
			neural_net->layers[lay_nb + 1]->neurons_out_act[i] = neural_net->activation_func(neural_net->layers[lay_nb + 1]->neurons_out[i]);
			i++;
		}
	}
	i = lay_nb + 2;
	while (i < neural_net->nb_layers)
	{
		update_layer(neural_net, i);
		i++;
	}
	partial_der = neural_net->cost_func(neural_net->layers[neural_net->nb_layers - 1]->neurons_out_act,
						exp_out, neural_net->layers[neural_net->nb_layers - 1]->nb_neurons);
	//if (lay_nb == 2)
	//	printf("ori_cost : %f\nnew_cost : %f\n", ori_cost, partial_der);
	partial_der = (partial_der - ori_cost) / neural_net->small_change;
	i = 0;
	if (lay_nb < neural_net->nb_layers - 1)
	{
		while (i < neural_net->layers[lay_nb + 1]->nb_neurons)
		{
			neural_net->layers[lay_nb + 1]->neurons_out[i] -= neural_net->layers[lay_nb]->neurons_out_act[neur_nb] * neural_net->layers[lay_nb + 1]->neurons[i][neur_nb];
			neural_net->layers[lay_nb + 1]->neurons_out[i] += ori_out_act * neural_net->layers[lay_nb + 1]->neurons[i][neur_nb];
			neural_net->layers[lay_nb + 1]->neurons_out_act[i] = neural_net->activation_func(neural_net->layers[lay_nb + 1]->neurons_out[i]);
			i++;
		}
	}
	i = lay_nb + 2;
	while (i < neural_net->nb_layers)
	{
		update_layer(neural_net, i);
		i++;
	}
	//neural_net->layers[lay_nb]->neurons_out[neur_nb] -= neural_net->layers[lay_nb - 1]->neurons_out_act[weight_nb] * neural_net->layers[lay_nb]->neurons[neur_nb][weight_nb];
	neural_net->layers[lay_nb]->neurons[neur_nb][weight_nb] = ori_val;
	neural_net->layers[lay_nb]->neurons_out_act[neur_nb] = ori_out_act;
	neural_net->layers[lay_nb]->neurons_out[neur_nb] = ori_out;
	//neural_net->layers[lay_nb]->neurons_out_act[neur_nb] = neural_net->activation_func(neural_net->layers[lay_nb]->neurons_out[neur_nb]);
	

	//neural_net->layers[lay_nb]->neurons_out[neur_nb] += neural_net->layers[lay_nb - 1]->neurons_out_act[weight_nb] * neural_net->layers[lay_nb]->neurons[neur_nb][weight_nb];
	//neural_net->layers[lay_nb]->neurons_out_act[neur_nb] = neural_net->activation_func(neural_net->layers[lay_nb]->neurons_out[neur_nb]);
	//printf("partial : %f\n", partial_der);
	return (partial_der);
}

void	update_gradient_with_input(t_neural_net *neural_net, double ***gradient, double *input, double *exp_out)
{
	double	ori_cost;
	double	*output;
	int		x;
	int		y;
	int		z;

	static int i = 0;
i++;
	output = get_output(neural_net, input);
	ori_cost = neural_net->cost_func(output, exp_out, neural_net->layers[neural_net->nb_layers - 1]->nb_neurons);
	//printf("ori_cost : %f\n", ori_cost);
	//if (i == 9)
	//	exit(0);
	x = 1;
	while (x < neural_net->nb_layers)
	{
		y = 0;
		while (y < neural_net->layers[x]->nb_neurons)
		{
			z = 0;
			while (z < neural_net->layers[x - 1]->nb_neurons + 1)
			{
				//if (x == 2)
				//	printf("gradientxyz 0 : %f\n", gradient[x][y][z]);
				gradient[x][y][z] += get_partial_derivative(neural_net, x, y, z, ori_cost, exp_out);
				//if (x == 2)
				//	printf("gradientxyz : %f\n", gradient[x][y][z]);
				z++;
			}
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

				//printf("%f\n", neural_net->layers[x]->neurons[y][z]);
				neural_net->layers[x]->neurons[y][z] += -(gradient[x][y][z] / mini_batch_size) * learning_rate;
				//printf("gradient : %f\n", -(gradient[x][y][z] / mini_batch_size));
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
			//printf("out : %f\n", out[x]);
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
			(void)learning_rate;
			update_weights(neural_net, gradient, mini_batch_size, learning_rate);
			x += mini_batch_size;
			//if (x == 60)
			//	exit(0);
		}
		i++;
		printf("EPOCH %d ; error rate : %.2f%%\n", i, score(neural_net, inputs, exp_outs, nb_inputs));
	}
}

/* ________________________________ END OF NN LIBRARY _______________________________*/

double	**get_inputs_from_images(t_image **images, int nb)
{
	int		i;
	double	**res;

	if (!(res = (double**)malloc(sizeof(double*) * nb)))
		malloc_error();
	i = 0;
	while (i < nb)
	{
		res[i] = images[i]->pixels;
		i++;
	}
	return (res);
}

double	**get_outputs_from_images(t_image **images, int nb, int nb2)
{
	int		i;
	int		x;
	double	**res;

	if (!(res = (double**)malloc(sizeof(double*) * nb)))
		malloc_error();
	i = 0;
	while (i < nb)
	{
		if (!(res[i] = (double*)malloc(sizeof(double) * nb2)))
			malloc_error();
		x = 0;
		while (x < nb2)
		{
			res[i][x] = 0.0;
			if (x == images[i]->label)
				res[i][x] = 1.0;
			x++;
		}
		i++;
	}
	return (res);
}

#define LEARNING_RATE 3.0
#define LAST_LAYER_NB_NEURONS 10
#define FISRT_LAYER_NB_NEURONS 784
#define NB_EPOCHS 30
#define MINI_BATCH_SIZE 10
#define COST_FUNC get_cost
#define ACTIVATION_FUNC sigmoid
#define NB_INPUTS 100
#define NB_INPUTS2 50000

int		main(void)
{
	t_image			**images;
	double			**inputs;
	double			**expected_outputs;
	double			**inputs2;
	double			**expected_outputs2;
	t_neural_net	*neural_net;
	int				nb_neurons_per_layer[] = {FISRT_LAYER_NB_NEURONS, 30, LAST_LAYER_NB_NEURONS};

	//srand(time(NULL));
	images = get_images();
	inputs = get_inputs_from_images(images, NB_INPUTS);
	expected_outputs = get_outputs_from_images(images, NB_INPUTS, LAST_LAYER_NB_NEURONS);
	inputs2 = get_inputs_from_images(images, NB_INPUTS2);
	expected_outputs2 = get_outputs_from_images(images, NB_INPUTS2, LAST_LAYER_NB_NEURONS);
	(void)images;
	(void)nb_neurons_per_layer;
	int *nb_ne = malloc(sizeof(int) * 3);
	nb_ne[0] = 784;
	nb_ne[1] = 30;
	nb_ne[2] = 10;
	neural_net = new_neural_net(nb_ne, 3, ACTIVATION_FUNC, COST_FUNC);
	(void)neural_net;
	//print_neural_network(neural_net);
	printf("ori_score : %.2f%%\n", score(neural_net, inputs, expected_outputs, NB_INPUTS));
	gradient_descent(neural_net, inputs, expected_outputs, NB_INPUTS, MINI_BATCH_SIZE, NB_EPOCHS, LEARNING_RATE);
	printf("score on 10000 : %.2f%%\n", score(neural_net, inputs, expected_outputs, NB_INPUTS));
	printf("score on 60000 : %.2f%%\n", score(neural_net, inputs2, expected_outputs2, NB_INPUTS2));
	//print_neural_network(neural_net);
	//save_neural_network_to_file(neural_net, "nn1.nn");
	return (0);
}