
/*
 * Gz.h - include file for the cs580 rendering library
 */

/*
 * universal constants
 */
#define GZ_SUCCESS      0
#define GZ_FAILURE      1

/*
 * name list tokens
 */
#define GZ_NULL_TOKEN			0    /* triangle vert attributes */
#define GZ_POSITION             1
#define GZ_NORMAL               2
#define GZ_TEXTURE_INDEX        3

/* renderer-state default pixel color */
#define GZ_RGB_COLOR            99	

#define GZ_INTERPOLATE			95		/* interpolation mode */

#define GZ_DIRECTIONAL_LIGHT	79	/* directional light */
#define GZ_AMBIENT_LIGHT		78	/* ambient light type */

#define GZ_AMBIENT_COEFFICIENT		1001	/* Ka material property */
#define GZ_DIFFUSE_COEFFICIENT		1002	/* Kd material property */
#define GZ_SPECULAR_COEFFICIENT		1003	/* Ks material property */
#define GZ_DISTRIBUTION_COEFFICIENT	1004	/* specular power of material */

#define	GZ_TEXTURE_MAP	1010		/* texture function ptr */

#include <cmath>

/*
 * value-list attributes
 */

/* select interpolation mode of the shader */
#define GZ_FLAT			0	/* do flat shading with GZ_RBG_COLOR */
#define	GZ_COLOR		1	/* interpolate vertex color */
#define	GZ_NORMALS		2	/* interpolate normals */

#ifndef GZVECTOR3D
#define GZVECTOR3D
typedef struct GzVector3D {
    float arr[3];

    // Default Constructor
    GzVector3D() {
        arr[0] = 0;
        arr[1] = 0;
        arr[2] = 0;
    }

    // Constructor of floats
    GzVector3D(float a, float b, float c) {

        arr[0] = a;
        arr[1] = b;
        arr[2] = c;
    }

    // Constructor of float arr
    GzVector3D(const float(&v)[3]) {

        arr[0] = v[0];
        arr[1] = v[1];
        arr[2] = v[2];
    }

    // Constructor of double arr
    GzVector3D(const double(&v)[3]) {

        arr[0] = (float)v[0];
        arr[1] = (float)v[1];
        arr[2] = (float)v[2];
    }

    // Destructor
    ~GzVector3D() {

    }

    // Copy constructor(deep copy)
    GzVector3D(const GzVector3D& other) {
        memcpy(arr, other.arr, sizeof(arr));
    }

    // Overloading [] operator for quick indexing
    float& operator[](int index) {
        return arr[index];
    }

    // Overloading + operator for Scalar add
    GzVector3D operator+(float s) const {
        return GzVector3D(
            arr[0] + s,
            arr[1] + s,
            arr[2] + s);
    }

    // Overloading + operator for Scalar add
    friend GzVector3D operator+(float s, const GzVector3D& v) {
        return v + s;
    }

    // Overloading + operator for vector add
    GzVector3D operator+(const GzVector3D& other) const {
        return GzVector3D(
            arr[0] + other.arr[0],
            arr[1] + other.arr[1],
            arr[2] + other.arr[2]);
    }

    // Overloading - operator for vector difference
    GzVector3D operator-(const GzVector3D& other) const {
        return GzVector3D(
            arr[0] - other.arr[0],
            arr[1] - other.arr[1],
            arr[2] - other.arr[2]);
    }

    // Overloading * operator for Dot product
    float operator*(const GzVector3D& other) const {
        return arr[0] * other.arr[0] + arr[1] * other.arr[1] + arr[2] * other.arr[2];
    }

    // Overloading * operator for Scalar muliply
    GzVector3D operator*(float s) const {
        return GzVector3D(
            arr[0] * s,
            arr[1] * s,
            arr[2] * s);
    }

    // Overloading * operator for Scalar muliply
    friend GzVector3D operator*(float s, const GzVector3D& v) {
        return v * s;
    }

    // Overloading * operator for Scalar divide
    GzVector3D operator/(float s) const {
        return GzVector3D(
            arr[0] / s,
            arr[1] / s,
            arr[2] / s);
    }

    // Overloading * operator for Scalar divide
    friend GzVector3D operator/(float s, const GzVector3D& v) {
        return v / s;
    }

    // Overloading ^ operator for Cross product
    GzVector3D operator^(const GzVector3D& other) const {
        return GzVector3D(
            arr[1] * other.arr[2] - arr[2] * other.arr[1],
            arr[2] * other.arr[0] - arr[0] * other.arr[2],
            arr[0] * other.arr[1] - arr[1] * other.arr[0]);
    }

    // Distance calculation
    float norm() {
        return sqrt(arr[0] * arr[0] + arr[1] * arr[1] + arr[2] * arr[2]);
    }

    // Return Normalized vector
    GzVector3D normalized() {
        float n = norm();
        return GzVector3D(
            arr[0] / n,
            arr[1] / n,
            arr[2] / n);
    }

    // Retrieve a copy of array in double
    double* GetDoubleArr() {
        static double output[3] = {
            (double)arr[0],
            (double)arr[1],
            (double)arr[2] };
        return output;
    }

    // Retrieve a copy of array in float
    float* GetFloatArr() {
        static float output[3] = {
            arr[0],
            arr[1],
            arr[2] };
        return output;
    }

} GzVector3D;
#endif

#ifndef GZVERTEX
#define GZVERTEX
typedef struct GzVertex
{
    // Source:https://en.wikipedia.org/wiki/List_of_refractive_indices
    // Use plate glass(window glass) refractive index as default
    float refract_index = (float)1.52;

    float position[3];
    float color_diffuse[3];
    float color_specular[3];
    float normal[3];
    float shininess;
    GzVertex(const float(&pos)[3], const float(&norm)[3]) {
        memcpy(position, pos, sizeof(position));
        memcpy(normal, norm, sizeof(normal));
        // color_diffuse 和 color_specular 添加默认值也可以
        memset(color_diffuse, 0, sizeof(color_diffuse));
        memset(color_specular, 0, sizeof(color_specular));
        shininess = 0;
    }
    GzVertex(float x, float y, float z) {
        position[0]=x;
        position[1]=y;
        position[2]=z;
    }
    GzVertex() {
        
    }
} GzVertex;
#endif

#ifndef GZTRIANGLE
#define GZTRIANGLE
typedef struct GzTriangle
{
    GzVertex v[3];

    GzTriangle(GzVertex v0, GzVertex v1, GzVertex v2) {
        v[0] = v0;
        v[1] = v1;
        v[2] = v2;
    }
    GzTriangle() {}
} GzTriangle;
#endif

#ifndef GZRAY
#define GZRAY
typedef struct GzRay {
  public:
    GzRay() {}
    GzRay(GzVector3D start, GzVector3D dir) {
        startPoint = start;
        direction = dir;
    };
    GzVector3D startPoint, direction;
} GzRay;
#endif

#ifndef GZSPHERE
#define GZSPHERE
typedef struct GzSphere
{
    double position[3];
    double color_diffuse[3];
    double color_specular[3];
    double shininess;
    double radius;
} GzSphere;
#endif

typedef int     GzToken;
typedef void    *GzPointer;
typedef float   GzColor[3];
typedef short   GzIntensity;	/* 0-4095 in lower 12-bits for RGBA */
typedef float	GzCoord[3];
typedef float	GzTextureIndex[2];
typedef float	GzMatrix[4][4];
typedef int	GzDepth;		/* signed z for clipping */

typedef	int	(*GzTexture)(float u, float v, GzColor color);	/* pointer to texture lookup method */
/* u,v parameters [0,1] are defined tex_fun(float u, float v, GzColor color) */

/*
 * Gz camera definition
 */
#ifndef GZCAMERA
#define GZCAMERA
typedef struct  GzCamera
{
  GzMatrix			Xiw;  	/* xform from world to image space */
  GzMatrix			Xpi;     /* perspective projection xform */
  GzCoord			position;  /* position of image plane origin */
  GzCoord			lookat;         /* position of look-at-point */
  GzCoord			worldup;   /* world up-vector (almost screen up) */
  float				FOV;            /* horizontal field of view */
} GzCamera;
#endif

#ifndef GZLIGHT
#define GZLIGHT
typedef struct  GzLight
{
  GzCoord        direction;    /* vector from surface to light */
  GzColor        color;		/* light color intensity */
} GzLight;
#endif

#ifndef GZINPUT
#define GZINPUT
typedef struct  GzInput
{
	GzCoord         rotation;       /* object rotation */
	GzCoord			translation;	/* object translation */
	GzCoord			scale;			/* object scaling */
	GzCamera		camera;			/* camera */
} GzInput;
#endif

#define RED     0         /* array indicies for color vector */
#define GREEN   1
#define BLUE    2

#define X       0      /* array indicies for position vector */
#define Y       1
#define Z       2

#define U       0       /* array indicies for texture coords */
#define V       1


#ifndef GZ_PIXEL
typedef	struct {
  GzIntensity    red;	
  GzIntensity    green;
  GzIntensity    blue;
  GzIntensity    alpha;
  GzDepth	 z;
} GzPixel;
#define GZ_PIXEL
#endif;

#define	MAXXRES	1024	/* put some bounds on size in case of error */
#define	MAXYRES	1024

