

#ifndef MNIST_READER_H
# define MNIST_READER_H

# include <fcntl.h>
# include <stdlib.h>
# include <unistd.h>
# include <stdio.h>

typedef struct	s_image
{
	double		*pixels;
	char		label;
}				t_image;

t_image			**get_images(void);

#endif