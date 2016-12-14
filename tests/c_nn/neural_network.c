

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

double	get_cost(double neural_net_output[], double expected_outputs[], int nb)
{
	double	val;
	double	ori_nb;

	val = 0.0;
	nb--;
	ori_nb = nb;
	while (nb >= 0)
	{
		val += (neural_net_output[nb] - expected_outputs[nb]) * (neural_net_output[nb] - expected_outputs[nb]);
		nb--;
	}
	return (val / ori_nb);
}

t_neuron	*new_neuron(int nb_weights_per_neuron)
{
	t_neuron	*res;

	if (!(res = (t_neuron*)malloc(sizeof(t_neuron))))
		malloc_error();
	if (!(res->weights = (double*)malloc(sizeof(double) * (nb_weights_per_neuron + 1))))
		malloc_error();
	res->nb_weights = nb_weights_per_neuron;
	res->value = 0.0;
	while (nb_weights_per_neuron >= 0)
	{
		res->weights[nb_weights_per_neuron] = ((float)rand()/(float)(RAND_MAX)) * 2.0 - 1.0;
		nb_weights_per_neuron--;
	}
	return (res);
}

t_layer		*new_layer(int nb_neurons, int nb_weights_per_neuron)
{
	t_layer		*res;

	if (!(res = (t_layer*)malloc(sizeof(t_layer))))
		malloc_error();
	res->nb_neurons = nb_neurons;
	if (!(res->neurons = (t_neuron**)malloc(sizeof(t_neuron*) * nb_neurons)))
		malloc_error();
	nb_neurons--;
	while (nb_neurons >= 0)
	{
		res->neurons[nb_neurons] = new_neuron(nb_weights_per_neuron);
		nb_neurons--;
	}
	return (res);
}

t_neural_net	*new_neural_net(int *nb_neurons_per_layer, int nb_layers, double learning_rate)
{
	t_neural_net	*res;
	int				i;

	if (!(res = (t_neural_net*)malloc(sizeof(t_neural_net))))
		malloc_error();
	if (!(res->layers = (t_layer**)malloc(sizeof(t_layer*) * nb_layers)))
		malloc_error();
	res->learning_rate = learning_rate;
	res->nb_layers = nb_layers;
	res->small_value = 0.1;
	i = 0;
	while (i < nb_layers)
	{
		res->layers[i] = (i == 0 ? new_layer(nb_neurons_per_layer[i], 0) : new_layer(nb_neurons_per_layer[i], nb_neurons_per_layer[i - 1]));
		i++;
	}
	return (res);
}

void	calculate_layer(t_neural_net *neural_net, int layer_nb, double (*activation_func)(double))
{
	int		i;

	i = 0;
	while (i < neural_net->layers[layer_nb]->nb_neurons)
	{
		calculate_neuron_value(neural_net->layers[layer_nb]->neurons[i], neural_net->layers[layer_nb - 1], activation_func);
		i++;
	}
}

double	*get_output(t_neural_net *neural_net, double (*activation_func)(double), double *ret)
{
	int		i;

	i = 1;
	while (i < neural_net->nb_layers)
	{
		calculate_layer(neural_net, i, activation_func);
		i++;
	}
	i = 0;
	while (i < neural_net->layers[neural_net->nb_layers - 1]->nb_neurons)
	{
		ret[i] = neural_net->layers[neural_net->nb_layers - 1]->neurons[i]->value;
		i++;
	}
	return (ret);
}

double	*update_output(t_neural_net *neural_net, double (*activation_func)(double), double *ret,
														int layer_nb, int neuron_nb)
{
	int		i;

	calculate_neuron_value(neural_net->layers[layer_nb]->neurons[neuron_nb], neural_net->layers[layer_nb - 1], activation_func);
	i = layer_nb + 1;
	while (i < neural_net->nb_layers)
	{
		calculate_layer(neural_net, i, activation_func);
		i++;
	}
	i = 0;
	while (i < neural_net->layers[neural_net->nb_layers - 1]->nb_neurons)
	{
		ret[i] = neural_net->layers[neural_net->nb_layers - 1]->neurons[i]->value;
		i++;
	}
	return (ret);
}

void	gradient_descent(t_neural_net *neural_net, double **inputs, double **expected_outputs, int nb_inputs,
										double (*cost_func)(double[], double[], int),
										double (*activation_func)(double), int nb_epochs, int mini_batch_size)
{
	int		i;
	int		x;
	double	***gradient;
	double	*ret;
	int		a;

	if (!(gradient = (double***)malloc(sizeof(double**) * (neural_net->nb_layers - 1))))
		malloc_error();
	i = 1;
	while (i < neural_net->nb_layers)
	{
		if (!(gradient[i - 1] = (double**)malloc(sizeof(double*) * neural_net->layers[i]->nb_neurons)))
			malloc_error();
		x = 0;
		while (x < neural_net->layers[i]->nb_neurons)
		{
			if (!(gradient[i - 1][x] = (double*)malloc(sizeof(double) * neural_net->layers[i]->neurons[x]->nb_weights)))
				malloc_error();
			x++;
		}
		i++;
	}
	if (!(ret = (double*)malloc(sizeof(double) * neural_net->layers[neural_net->nb_layers - 1]->nb_neurons)))
		malloc_error();
	ret = get_output(neural_net, activation_func, ret);
	a = 0;
	while (a < nb_epochs)
	{
		i = 0;
		while (i < nb_inputs)
		{
			//printf("input n %d\n", i);
			x = i;
			while (x < i + mini_batch_size)
			{
				set_first_layer(neural_net, inputs[x]);
				get_gradient(neural_net, cost_func, activation_func, expected_outputs[x], ret, gradient);
				update_weights(neural_net, gradient);
				x++;
			}
			i += mini_batch_size;
		}
		//if (a % 50 == 0)
		//{
			printf("EPOCH %d ; ", a);
			score(neural_net, inputs, expected_outputs, nb_inputs, activation_func);
		//}
		a++;
	}
}

void	set_first_layer(t_neural_net *neural_net, double *input)
{
	int 	i;

	i = 0;
	while (i < neural_net->layers[0]->nb_neurons)
	{
		neural_net->layers[0]->neurons[i]->value = input[i];
		i++;
	}
}

void	calculate_neuron_value(t_neuron *neuron, t_layer *previous_layer, double (*activation_func)(double))
{
	int		i;

	if (neuron->nb_weights == 0)
		return ;
	neuron->value = -1.0 * neuron->weights[0];
	i = 1;
	while (i + 10 <= neuron->nb_weights)
	{
		neuron->value += previous_layer->neurons[i - 1]->value * neuron->weights[i] + 
							previous_layer->neurons[i]->value * neuron->weights[i + 1] +
							previous_layer->neurons[i + 1]->value * neuron->weights[i + 2] +
							previous_layer->neurons[i + 2]->value * neuron->weights[i + 3] +
							previous_layer->neurons[i + 3]->value * neuron->weights[i + 4] +
							previous_layer->neurons[i + 4]->value * neuron->weights[i + 5] +
							previous_layer->neurons[i + 5]->value * neuron->weights[i + 6] +
							previous_layer->neurons[i + 6]->value * neuron->weights[i + 7] +
							previous_layer->neurons[i + 7]->value * neuron->weights[i + 8] +
							previous_layer->neurons[i + 8]->value * neuron->weights[i + 9];
		i += 10;
	}
	while (i < neuron->nb_weights)
	{
		neuron->value += previous_layer->neurons[i - 1]->value * neuron->weights[i];
		i++;;
	}
	neuron->value = activation_func(neuron->value);
}

double	*predict(t_neural_net *neural_net, double *input, double (*activation_func)(double))
{
	double	*ret;

	if (!(ret = (double*)malloc(sizeof(double) * neural_net->layers[neural_net->nb_layers - 1]->nb_neurons)))
		malloc_error();
	set_first_layer(neural_net, input);
	return (get_output(neural_net, activation_func, ret));
}

void	update_weights(t_neural_net *neural_net, double ***gradient)
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
			while (z < neural_net->layers[x]->neurons[y]->nb_weights)
			{
				neural_net->layers[x]->neurons[y]->weights[z] += -gradient[x - 1][y][z] * neural_net->learning_rate;
				//if (x == 2 && y == 5)
				//	printf("%f\n", gradient[x - 1][y][z]);
				z++;
			}
			y++;
		}
		x++;
	}
}

double	get_partial_der(t_neural_net *neural_net, int layer_nb, int neuron_nb, int weight_nb,
										double (*cost_func)(double[], double[], int),
										double (*activation_func)(double), double *expected_output,
															double ori_cost, double *ret)
{
	double	ori_val;
	double	cost;

	ori_val = neural_net->layers[layer_nb]->neurons[neuron_nb]->weights[weight_nb];
	neural_net->layers[layer_nb]->neurons[neuron_nb]->weights[weight_nb] += neural_net->small_value;
	update_output(neural_net, activation_func, ret, layer_nb, neuron_nb);
	cost = cost_func(ret, expected_output, neural_net->layers[neural_net->nb_layers - 1]->nb_neurons);
	neural_net->layers[layer_nb]->neurons[neuron_nb]->weights[weight_nb] = ori_val;
	update_output(neural_net, activation_func, ret, layer_nb, neuron_nb);
	return ((cost - ori_cost) / neural_net->small_value);
}

void	get_gradient(t_neural_net *neural_net, double (*cost_func)(double[], double[], int),
										double (*activation_func)(double), double *expected_output,
															double *ret, double ***gradient)
{
	int		x;
	int		y;
	int		z;
	double	ori_cost;

	get_output(neural_net, activation_func, ret);
	ori_cost = cost_func(ret, expected_output, neural_net->layers[neural_net->nb_layers - 1]->nb_neurons);
	x = 1;
	while (x < neural_net->nb_layers)
	{
		y = 0;
		while (y < neural_net->layers[x]->nb_neurons)
		{
			z = 0;
			//printf("layer : %d, neuron : %d\n", x, y);
			while (z < neural_net->layers[x]->neurons[y]->nb_weights)
			{
				gradient[x - 1][y][z] = get_partial_der(neural_net, x, y, z, cost_func,
										activation_func, expected_output, ori_cost, ret);
				z++;
			}
			y++;
		}
		x++;
	}
}

void	score(t_neural_net *neural_net, double **inputs, double **expected_outputs, int nb_inputs,
																double (*activation_func)(double))
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
		out = predict(neural_net, inputs[i], activation_func);
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
	//printf("got %d correct.\ngot %d incorrect\n", correct, incorrect);
	printf("error rate : %.2f%%\n", (float)incorrect / (correct + incorrect) * 100.0);
}

#define LEARNING_RATE 3.0
#define LAST_LAYER_NB_NEURONS 10
#define FISRT_LAYER_NB_NEURONS 784
#define NB_EPOCHS 20
#define MINI_BATCH_SIZE 10
#define COST_FUNC get_cost
#define ACTIVATION_FUNC sigmoid
#define NB_INPUTS 100
#define NB_INPUTS2 50000

void	print_neural_network(t_neural_net *neural_net)
{
	int		i;

	i = 0;
	while (i < neural_net->layers[2]->neurons[5]->nb_weights)
	{
		printf("%f ", neural_net->layers[2]->neurons[5]->weights[i]);
		i++;
	}
	printf("\n");
}

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

void	save_neural_network_to_file(t_neural_net *neural_net, char *filename)
{
	int		fd;
	int		x;
	int		y;
	int		z;
	char	output[50];
	int		lol;

	fd = open(filename, O_RDWR | O_CREAT, S_IRWXU | S_IRWXG | S_IRWXO);
	x = 1;
	while (x < neural_net->nb_layers)
	{
		y = 0;
		while (y < neural_net->layers[x]->nb_neurons)
		{
			z = 0;
			while (z < neural_net->layers[x]->neurons[y]->nb_weights)
			{
				snprintf(output, 50, "%f", neural_net->layers[x]->neurons[y]->weights[z]);
				lol = write(fd, output, strlen(output));
				z++;
				lol = write(fd, ";", 1);
			}
			y++;
			lol = write(fd, "\t", 1);
		}
		x++;
		lol = write(fd, "\n", 1);
	}
	(void)lol;
}

int		main(void)
{
	t_image			**images;
	double			**inputs;
	double			**expected_outputs;
	double			**inputs2;
	double			**expected_outputs2;
	t_neural_net	*neural_net;
	int				nb_neurons_per_layer[] = {FISRT_LAYER_NB_NEURONS, 30, LAST_LAYER_NB_NEURONS};

	srand(time(NULL));
	images = get_images();
	inputs = get_inputs_from_images(images, NB_INPUTS);
	expected_outputs = get_outputs_from_images(images, NB_INPUTS, LAST_LAYER_NB_NEURONS);
	inputs2 = get_inputs_from_images(images, NB_INPUTS2);
	expected_outputs2 = get_outputs_from_images(images, NB_INPUTS2, LAST_LAYER_NB_NEURONS);
	neural_net = new_neural_net(nb_neurons_per_layer, 3, LEARNING_RATE);
	print_neural_network(neural_net);
	score(neural_net, inputs, expected_outputs, NB_INPUTS, ACTIVATION_FUNC);
	gradient_descent(neural_net, inputs, expected_outputs, NB_INPUTS, COST_FUNC, ACTIVATION_FUNC, NB_EPOCHS, MINI_BATCH_SIZE);
	score(neural_net, inputs, expected_outputs, NB_INPUTS, ACTIVATION_FUNC);
	score(neural_net, inputs2, expected_outputs2, NB_INPUTS2, ACTIVATION_FUNC);
	print_neural_network(neural_net);
	save_neural_network_to_file(neural_net, "nn1.nn");
	return (0);
}