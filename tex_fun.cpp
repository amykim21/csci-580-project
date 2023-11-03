/* Texture functions for cs580 GzLib	*/
#include    "stdafx.h" 
#include	"stdio.h"
#include	"Gz.h"
#include <complex>
GzColor	*image=NULL;
int xs, ys;
int reset = 1;

/* Image texture function */
int tex_fun(float u, float v, GzColor color)
{
  unsigned char		pixel[3];
  unsigned char     dummy;
  char  		foo[8];
  int   		i, j;
  FILE			*fd;

  if (reset) {          /* open and load texture file */
    fd = fopen ("texture", "rb");
    if (fd == NULL) {
      fprintf (stderr, "texture file not found\n");
      exit(-1);
    }
    fscanf (fd, "%s %d %d %c", foo, &xs, &ys, &dummy);
    image = (GzColor*)malloc(sizeof(GzColor)*(xs+1)*(ys+1));
    if (image == NULL) {
      fprintf (stderr, "malloc for texture image failed\n");
      exit(-1);
    }

    for (i = 0; i < xs*ys; i++) {	/* create array of GzColor values */
      fread(pixel, sizeof(pixel), 1, fd);
      image[i][RED] = (float)((int)pixel[RED]) * (1.0 / 255.0);
      image[i][GREEN] = (float)((int)pixel[GREEN]) * (1.0 / 255.0);
      image[i][BLUE] = (float)((int)pixel[BLUE]) * (1.0 / 255.0);
      }

    reset = 0;          /* init is done */
	fclose(fd);
  }

/* bounds-test u,v to make sure nothing will overflow image array bounds */
/* determine texture cell corner values and perform bilinear interpolation */
/* set color to interpolated GzColor value and return */
  int ul, u2, vl, v2;
  GzColor lerpVals;
  if (u < 0)
      u = 0;
  else if (u > 1)
      u = 1;

  if (v < 0)
      v = 0;
  else if (v > 1)
      v = 1;
  //Check for bounds
  u *= xs;
  v *= ys;
  u2 = (int)(u + 1.0f);
  v2 = (int)(v + 1.0f);
  ul = (int)u;
  vl = (int)v;
  if (u2 >= xs)
      u2 = xs - 1;
  if (v2 >= ys)
      v2 = ys - 1;
  //Bilinear Interpolation
  float s = u - (float)ul;
  float t = v - (float)vl;
  for (int i = 0; i < 3; i++) {
      lerpVals[i] = (1 - s) * (1 - t) * image[ul + vl * xs][i] +
          s * (1 - t) * image[u2 + vl * xs][i] +
          (1 - s) * t * image[ul + v2 * xs][i] +
          s * t * image[u2 + v2 * xs][i];
  }
  //Assigning Color Values
  for (int i = 0; i < 3; i++)
      color[i] = lerpVals[i];
	return GZ_SUCCESS;
}

#define NOISE_INTENS 0.1f
float interpolate_pre(float a, float b, float x)
{
    float ft = x * 3.1415927;
    float f = (1 - cos(ft)) * 0.5;

    return  a * (1 - f) + b * f;
}
float noise(int x, int y)
{
    int n = x + y * 57;
    n = (n << 13) ^ n;
    return (1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0);
}

float interpolated_noise(float x, float y)
{
    int integer_X = int(x);
    float fractional_X = x - integer_X;

    int integer_Y = int(y);
    float fractional_Y = y - integer_Y;

    float v1 = noise(integer_X, integer_Y);
    float v2 = noise(integer_X + 1, integer_Y);
    float v3 = noise(integer_X, integer_Y + 1);
    float v4 = noise(integer_X + 1, integer_Y + 1);

    float i1 = interpolate_pre(v1, v2, fractional_X);
    float i2 = interpolate_pre(v3, v4, fractional_X);

    return interpolate_pre(i1, i2, fractional_Y);
}


#define SIZE 8 // 格子尺寸

int ptex_fun(float u, float v, GzColor color)
{
    /* 将u, v的范围映射到0到SIZE */
    int x = (int)(u * SIZE);
    int y = (int)(v * SIZE);

    /* 检查当前的图块是黑还是白 */
    if ((x + y) % 2 == 0) {
        /* 黑色图块 */
        color[0] = 0.0f;
        color[1] = 0.0f;
        color[2] = 0.0f;
    }
    else {
        /* 白色图块 */
        color[0] = 1.0f;
        color[1] = 1.0f;
        color[2] = 1.0f;
    }

    return GZ_SUCCESS;
}



/* Free texture memory */
int GzFreeTexture()
{
	if(image!=NULL)
		free(image);
	return GZ_SUCCESS;
}

