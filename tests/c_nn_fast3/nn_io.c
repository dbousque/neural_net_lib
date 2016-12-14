

#include "neural_network.h"

static void	malloc_error(void)
{
	printf("allocation failed.\n");
	exit(1);
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

double	*decal_one_right(double *inp, int inp_size)
{
	int		i;
	double	*res;

	if (!(res = (double*)malloc(sizeof(double) * inp_size)))
		malloc_error();
	i = 1;
	while (i < inp_size)
	{
		res[i] = inp[i - 1];
		i++;
	}
	res[0] = inp[inp_size - 1];
	return (res);
}

double	*decal_one_left(double *inp, int inp_size)
{
	int		i;
	double	*res;

	if (!(res = (double*)malloc(sizeof(double) * inp_size)))
		malloc_error();
	i = 0;
	while (i < inp_size - 1)
	{
		res[i] = inp[i + 1];
		i++;
	}
	res[inp_size - 1] = inp[0];
	return (res);
}

double	*decal_one_down(double *inp, int row_size)
{
	int		i;
	int		x;
	double	*res;

	if (!(res = (double*)malloc(sizeof(double) * (row_size * row_size))))
		malloc_error();
	i = row_size;
	while (i < row_size * row_size)
	{
		res[i] = inp[i - row_size];
		i++;
	}
	i = 0;
	x = row_size * row_size - row_size;
	while (x < row_size * row_size)
	{
		res[i] = inp[x];
		x++;
		i++;
	}
	return (res);
}

double	*decal_one_up(double *inp, int row_size)
{
	int		i;
	int		x;
	double	*res;

	if (!(res = (double*)malloc(sizeof(double) * (row_size * row_size))))
		malloc_error();
	i = 0;
	while (i < row_size * row_size - row_size)
	{
		res[i] = inp[i + row_size];
		i++;
	}
	i = 0;
	x = row_size * row_size - row_size;
	while (x < row_size * row_size)
	{
		res[x] = inp[i];
		x++;
		i++;
	}
	return (res);
}

void	generate_distortions(double ***inputs, double ***outputs, int nb)
{
	double	**res_inputs;
	double	**res_outputs;
	int		i;

	if (!(res_inputs = (double**)malloc(sizeof(double*) * (nb * 5))))
		malloc_error();
	if (!(res_outputs = (double**)malloc(sizeof(double*) * (nb * 5))))
		malloc_error();
	i = 0;
	while (i < nb)
	{
		res_inputs[i] = (*inputs)[i];
		res_outputs[i] = (*inputs)[i];
		i++;
	}
	i = 0;
	while (i < nb)
	{
		res_inputs[i + nb] = decal_one_right((*inputs)[i], 28 * 28);
		res_outputs[i + nb] = (*outputs)[i];
		i++;
	}
	i = 0;
	while (i < nb)
	{
		res_inputs[i + (nb * 2)] = decal_one_left((*inputs)[i], 28 * 28);
		res_outputs[i + (nb * 2)] = (*outputs)[i];
		i++;
	}
	i = 0;
	while (i < nb)
	{
		res_inputs[i + (nb * 3)] = decal_one_down((*inputs)[i], 28);
		res_outputs[i + (nb * 3)] = (*outputs)[i];
		i++;
	}
	i = 0;
	while (i < nb)
	{
		res_inputs[i + (nb * 4)] = decal_one_up((*inputs)[i], 28);
		res_outputs[i + (nb * 4)] = (*outputs)[i];
		i++;
	}
	free(*inputs);
	free(*outputs);
	*inputs = res_inputs;
	*outputs = res_outputs;
}

void	write_image_to_file(double *image, char *filename)
{
	int				fd;
	int				i;
	int				header[122];
	unsigned char	c;
	int				lol;

	header[0] = 73;
	header[1] = 73;
	header[2] = 42;
	header[3] = 0;
	header[4] = 8;
	header[5] = 0;
	header[6] = 0;
	header[7] = 0;
	header[8] = 9;
	header[9] = 0;
	header[10] = 0;
	header[11] = 1;
	header[12] = 3;
	header[13] = 0;
	header[14] = 1;
	header[15] = 0;
	header[16] = 0;
	header[17] = 0;
	header[18] = 28;
	header[19] = 0;
	header[20] = 0;
	header[21] = 0;
	header[22] = 1;
	header[23] = 1;
	header[24] = 3;
	header[25] = 0;
	header[26] = 1;
	header[27] = 0;
	header[28] = 0;
	header[29] = 0;
	header[30] = 28;
	header[31] = 0;
	header[32] = 0;
	header[33] = 0;
	header[34] = 2;
	header[35] = 1;
	header[36] = 3;
	header[37] = 0;
	header[38] = 1;
	header[39] = 0;
	header[40] = 0;
	header[41] = 0;
	header[42] = 8;
	header[43] = 0;
	header[44] = 0;
	header[45] = 0;
	header[46] = 3;
	header[47] = 1;
	header[48] = 3;
	header[49] = 0;
	header[50] = 1;
	header[51] = 0;
	header[52] = 0;
	header[53] = 0;
	header[54] = 1;
	header[55] = 0;
	header[56] = 0;
	header[57] = 0;
	header[58] = 6;
	header[59] = 1;
	header[60] = 3;
	header[61] = 0;
	header[62] = 1;
	header[63] = 0;
	header[64] = 0;
	header[65] = 0;
	header[66] = 1;
	header[67] = 0;
	header[68] = 0;
	header[69] = 0;
	header[70] = 17;
	header[71] = 1;
	header[72] = 4;
	header[73] = 0;
	header[74] = 1;
	header[75] = 0;
	header[76] = 0;
	header[77] = 0;
	header[78] = 122;
	header[79] = 0;
	header[80] = 0;
	header[81] = 0;
	header[82] = 22;
	header[83] = 1;
	header[84] = 3;
	header[85] = 0;
	header[86] = 1;
	header[87] = 0;
	header[88] = 0;
	header[89] = 0;
	header[90] = 28;
	header[91] = 0;
	header[92] = 0;
	header[93] = 0;
	header[94] = 23;
	header[95] = 1;
	header[96] = 3;
	header[97] = 0;
	header[98] = 1;
	header[99] = 0;
	header[100] = 0;
	header[101] = 0;
	header[102] = 16;
	header[103] = 3;
	header[104] = 0;
	header[105] = 0;
	header[106] = 28;
	header[107] = 1;
	header[108] = 3;
	header[109] = 0;
	header[110] = 1;
	header[111] = 0;
	header[112] = 0;
	header[113] = 0;
	header[114] = 1;
	header[115] = 0;
	header[116] = 0;
	header[117] = 0;
	header[118] = 0;
	header[119] = 0;
	header[120] = 0;
	header[121] = 0;
	fd = open(filename, O_RDWR | O_CREAT, S_IRWXU | S_IRWXG | S_IRWXO);
	i = 0;
	while (i < 122)
	{
		lol = write(fd, &(header[i]), 1);
		i++;
	}
	i = 0;
	while (i < 28 * 28)
	{
		c = ((int)(image[i] * 255.0));
		lol = write(fd, &c, 1);
		i++;
	}
	(void)lol;
	close(fd);
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
			while (z < neural_net->layers[x - 1]->nb_neurons + 1)
			{
				snprintf(output, 50, "%f", neural_net->layers[x]->neurons[y][z]);
				lol = write(fd, output, strlen(output));
				z++;
				lol = write(fd, ";", 1);
			}
			y++;
			lol = write(fd, "\n", 1);
		}
		x++;
		lol = write(fd, "\n", 1);
	}
	(void)lol;
}