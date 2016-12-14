

#include "mnist_reader.h"

static void	malloc_error(void)
{
	printf("allocation failed.\n");
	exit(1);
}

char	*read_whole_file(char *filename, int *size_ret)
{
	char  *res;
	char  buf[1024];
	long  size;
	int   ret;
	int   fd;

	fd = open(filename, O_RDONLY);
	if (fd < 0)
	{
		printf("ERROR WHILE OPENING FILE !\n");
		exit(0);
	}
	size = 0;
	while ((ret = read(fd, buf, 1024)) > 0)
		size += ret;
	close(fd);
	fd = open(filename, O_RDONLY);
	if (fd < 0)
	{
    	printf("ERROR WHILE OPENING FILE 2 !\n");
    	exit(0);
	}
	*size_ret = size;
	res = (char*)malloc(sizeof(char) * (size + 1));
	int lol = read(fd, res, size);
	(void)lol;
	res[size] = '\0';
	close(fd);
	return (res);
}

t_image	**get_images(void)
{
	char	*file_content;
	t_image	**images;
	int		i;
	int		x;
	int		ind;
	int		file_size;

	if (!(images = (t_image**)malloc(sizeof(t_image*) * 60000)))
		malloc_error();
	file_content = read_whole_file("train-images.idx3-ubyte", &file_size);
	i = 0;
	ind = 16;
	while (ind < file_size)
	{
		x = 0;
		if (!(images[i] = (t_image*)malloc(sizeof(t_image))))
			malloc_error();
		if (!(images[i]->pixels = (double*)malloc(sizeof(double) * 28 * 28)))
			malloc_error();
		while (x < 28 * 28)
		{
			images[i]->pixels[x] = ((double)((unsigned char)file_content[ind + x])) / 255.0;
			x++;
		}
		ind += 28 * 28;
		i++;
	}
	free(file_content);
	file_content = read_whole_file("train-labels.idx1-ubyte", &file_size);
	i = 8;
	while (i < 60000 + 8)
	{
		images[i - 8]->label = file_content[i];
		i++;
	}
	free(file_content);
	return (images);
}

void	print_image(t_image *image)
{
	int		i;

	i = 0;
	while (i < 28 * 28)
	{
		printf("%f\n", image->pixels[i]);
		i++;
	}
	printf("label : %d\n", image->label);
}