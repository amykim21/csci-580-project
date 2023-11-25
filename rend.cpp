
/* CS580 Homework 4 */

#include	"stdafx.h"
#include	"stdio.h"
#include	"math.h"
#include	"Gz.h"
#include <algorithm>
#include <vector>
#include <cmath>
#include <stdlib.h> 
#include <limits>
#include <cfloat>

#include	"rend.h"

#define PI (float) 3.14159265358979323846
#define DEG2RAD(degree) ((degree) * (PI / 180.0))
typedef std::vector<float> VectorCoord;
typedef std::vector<std::vector<float>> VectorMatrix;

/**
class BSPTree {
public:
    BSPNode* root;

    BSPTree() : root(nullptr) {}

    ~BSPTree() {
        delete root;
    }

    // Recursively build the BSP tree from a list of objects
    BSPNode* buildTree(const std::vector<Object*>& objects, int depth = 0) {
        if (objects.empty() || depth > MAX_DEPTH) {
            return nullptr;
        }

        BSPNode* node = new BSPNode();

        Plane partitionPlane = choosePartitionPlane(objects);
        std::vector<Object*> frontObjects;
        std::vector<Object*> backObjects;
        partitionObjects(objects, partitionPlane, frontObjects, backObjects);

        node->partitionPlane = partitionPlane;
        node->front = buildTree(frontObjects, depth + 1);
        node->back = buildTree(backObjects, depth + 1);

        // If this is a leaf node, assign the objects to this node
        if (node->isLeaf()) {
            node->objects = objects;
        }

        return node;
    }

    // Traverse the BSP tree with a ray to find the closest intersection
    Object* traverse(Ray& ray, BSPNode* node, float& closestDistance) {
        if (!node || node->isLeaf()) {
            return findClosestIntersection(ray, node->objects, closestDistance);
        }

        // Determine the order to traverse front and back child based on the ray direction
        bool frontFirst = ray.direction.dot(node->partitionPlane.normal) < 0;
        BSPNode* firstChild = frontFirst ? node->front : node->back;
        BSPNode* secondChild = frontFirst ? node->back : node->front;

        // Traverse the first child
        Object* closestObject = traverse(ray, firstChild, closestDistance);

        // If the closest intersection is further than the partition plane, also check the second child
        if (closestDistance > node->partitionPlane.distance(ray.origin)) {
            Object* secondClosestObject = traverse(ray, secondChild, closestDistance);
            if (secondClosestObject) {
                closestObject = secondClosestObject;
            }
        }

        return closestObject;
    }

private:
    // A function to choose the best partition plane based on a heuristic
    Plane choosePartitionPlane(const std::vector<Object*>& objects) {
        // Implement heuristic here
    }

    // A function to partition objects into front and back lists based on a plane
    void partitionObjects(const std::vector<Object*>& objects, const Plane& plane,
                          std::vector<Object*>& frontObjects, std::vector<Object*>& backObjects) {
        // Implement partitioning logic here
    }

    // A function to find the closest intersection within a list of objects
    Object* findClosestIntersection(Ray& ray, const std::vector<Object*>& objects, float& closestDistance) {
        // Implement intersection logic here
    }
};
*/

/*
BSPTree bspTree;
bspTree.root = bspTree.buildTree(sceneObjects);
float closestDistance = std::numeric_limits<float>::infinity();
Object* closestObject = bspTree.traverse(ray, bspTree.root, closestDistance);
*/
int GzRender::GzRotXMat(float degree, GzMatrix mat)
{
	/* HW 3.1
	// Create rotate matrix : rotate along x axis
	// Pass back the matrix using mat value
	*/
	degree = DEG2RAD(degree);

	mat[0][0] = 1.0;
	mat[0][1] = 0.0;
	mat[0][2] = 0.0;
	mat[0][3] = 0.0;

	mat[1][0] = 0.0;
	mat[1][1] = cos(degree);
	mat[1][2] = -sin(degree);
	mat[1][3] = 0.0;

	mat[2][0] = 0.0;
	mat[2][1] = sin(degree);
	mat[2][2] = cos(degree);
	mat[2][3] = 0.0;

	mat[3][0] = 0.0;
	mat[3][1] = 0.0;
	mat[3][2] = 0.0;
	mat[3][3] = 1.0;

	return GZ_SUCCESS;
}

int GzRender::GzRotYMat(float degree, GzMatrix mat)
{
	/* HW 3.2
	// Create rotate matrix : rotate along y axis
	// Pass back the matrix using mat value
	*/
	degree = DEG2RAD(degree);

	mat[0][0] = cos(degree);
	mat[0][1] = 0.0;
	mat[0][2] = sin(degree);
	mat[0][3] = 0.0;

	mat[1][0] = 0.0;
	mat[1][1] = 1.0;
	mat[1][2] = 0.0;
	mat[1][3] = 0.0;

	mat[2][0] = -sin(degree);
	mat[2][1] = 0.0;
	mat[2][2] = cos(degree);
	mat[2][3] = 0.0;

	mat[3][0] = 0.0;
	mat[3][1] = 0.0;
	mat[3][2] = 0.0;
	mat[3][3] = 1.0;
	return GZ_SUCCESS;
}

int GzRender::GzRotZMat(float degree, GzMatrix mat)
{
	/* HW 3.3
	// Create rotate matrix : rotate along z axis
	// Pass back the matrix using mat value
	*/
	degree = DEG2RAD(degree);

	mat[0][0] = cos(degree);
	mat[0][1] = -sin(degree);
	mat[0][2] = 0.0;
	mat[0][3] = 0.0;

	mat[1][0] = sin(degree);
	mat[1][1] = cos(degree);
	mat[1][2] = 0.0;
	mat[1][3] = 0.0;

	mat[2][0] = 0.0;
	mat[2][1] = 0.0;
	mat[2][2] = 1.0;
	mat[2][3] = 0.0;

	mat[3][0] = 0.0;
	mat[3][1] = 0.0;
	mat[3][2] = 0.0;
	mat[3][3] = 1.0;
	return GZ_SUCCESS;
}

int GzRender::GzTrxMat(GzCoord translate, GzMatrix mat)
{
	/* HW 3.4
	// Create translation matrix
	// Pass back the matrix using mat value
	*/
	// Create translation matrix
	mat[0][0] = 1.0f;  mat[0][1] = 0.0f;  mat[0][2] = 0.0f;  mat[0][3] = translate[0];
	mat[1][0] = 0.0f;  mat[1][1] = 1.0f;  mat[1][2] = 0.0f;  mat[1][3] = translate[1];
	mat[2][0] = 0.0f;  mat[2][1] = 0.0f;  mat[2][2] = 1.0f;  mat[2][3] = translate[2];
	mat[3][0] = 0.0f;  mat[3][1] = 0.0f;  mat[3][2] = 0.0f;  mat[3][3] = 1.0f;

	return GZ_SUCCESS;
}

int GzRender::GzScaleMat(GzCoord scale, GzMatrix mat)
{
	/* HW 3.5
	// Create scaling matrix
	// Pass back the matrix using mat value
	*/
	// Create translation matrix
	mat[0][0] = scale[0];  mat[0][1] = 0.0f;      mat[0][2] = 0.0f;      mat[0][3] = 0.0f;
	mat[1][0] = 0.0f;      mat[1][1] = scale[1];  mat[1][2] = 0.0f;      mat[1][3] = 0.0f;
	mat[2][0] = 0.0f;      mat[2][1] = 0.0f;      mat[2][2] = scale[2];  mat[2][3] = 0.0f;
	mat[3][0] = 0.0f;      mat[3][1] = 0.0f;      mat[3][2] = 0.0f;      mat[3][3] = 1.0f;
	

	return GZ_SUCCESS;
}

GzRender::GzRender(int xRes, int yRes)
{
	/* HW1.1 create a framebuffer for MS Windows display:
	 -- set display resolution
	 -- allocate memory for framebuffer : 3 bytes(b, g, r) x width x height
	 -- allocate memory for pixel buffer
	 */
	xres = (short)xRes;
	yres = (short)yRes;
	int totalPixels = xRes * yRes;
	pixelbuffer = new GzPixel[totalPixels];
	framebuffer = new char[3 * totalPixels];
	numlights = 0;
	/* HW 3.6
	- setup Xsp and anything only done once
	- init default camera
	*/
	m_camera.lookat[X] = 0;
	m_camera.lookat[Y] = 0;
	m_camera.lookat[Z] = 0;
	m_camera.position[X] = DEFAULT_IM_X;
	m_camera.position[Y] = DEFAULT_IM_Y;
	m_camera.position[Z] = DEFAULT_IM_Z;

	m_camera.worldup[X] = 0;
	m_camera.worldup[Y] = 1;
	m_camera.worldup[Z] = 0;
	m_camera.FOV = DEFAULT_FOV;
}

GzRender::~GzRender()
{
	/* HW1.2 clean up, free buffer memory */
	delete[] pixelbuffer;
	delete[] framebuffer;
}

int GzRender::GzDefault()
{
	/* HW1.3 set pixel buffer to some default values - start a new frame */
	if (!framebuffer || !pixelbuffer) return GZ_FAILURE;

	// reset framebuffer and pixelbuffer
	for (int i = 0; i < xres * yres * 3; ++i)
		framebuffer[i] = 0;
	for (int i = 0; i < xres * yres; ++i)
		pixelbuffer[i] = { 128 << 4, 112 << 4, 96 << 4, 1,MAXINT };
	return GZ_SUCCESS;
}

float norm(VectorCoord v) {
	return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

float norm(GzCoord v) {
	return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

VectorCoord normalize(VectorCoord v) {
	float len = norm(v);
	VectorCoord result = {
		v[0] / len,
		v[1] / len,
		v[2] / len
	};
	return result;
}

VectorCoord normalize(GzCoord v) {
	float len = norm(v);
	VectorCoord result = {
		v[0] / len,
		v[1] / len,
		v[2] / len
	};
	return result;
}

float dot(VectorCoord v1, VectorCoord v2) {
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

float dot(GzCoord v1, GzCoord v2) {
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

float dot(VectorCoord v1, GzCoord v2) {
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

VectorCoord multiply(float v1, VectorCoord v2) {
	VectorCoord result = {
		v1 * v2[0],
		v1 * v2[1],
		v1 * v2[2]
	};
	return result;
}

VectorCoord cross(VectorCoord v1, VectorCoord v2) {
	VectorCoord result = {
		v1[1] * v2[2] - v1[2] * v2[1],
		v1[2] * v2[0] - v1[0] * v2[2],
		v1[0] * v2[1] - v1[1] * v2[0]
	};
	return result;
}

VectorCoord decreaseCoord(VectorCoord v1, VectorCoord v2) {
	VectorCoord result = {
		v1[0] - v2[0],
		v1[1] - v2[1],
		v1[2] - v2[2],
	};
	return result;
}

VectorMatrix multiplyMatrix(VectorMatrix m1, VectorMatrix m2) {
	VectorMatrix result(4, std::vector<float>(4, 0));
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			result[i][j] = 0;
			for (int k = 0; k < 4; ++k) {
				result[i][j] += m1[i][k] * m2[k][j];
			}
		}
	}
	return result;
}

int GzRender::GzBeginRender()
{
	/* HW 3.7
	- setup for start of each frame - init frame buffer color,alpha,z
	- compute Xiw and projection xform Xpi from camera definition
	- init Ximage - put Xsp at base of stack, push on Xpi and Xiw
	- now stack contains Xsw and app can push model Xforms when needed
	*/

	VectorCoord camDir, camRight, camUp;


	VectorCoord vecLookAt(m_camera.lookat, m_camera.lookat + sizeof m_camera.lookat / sizeof m_camera.lookat[0]);
	VectorCoord vecPosition(m_camera.position, m_camera.position + sizeof m_camera.position / sizeof m_camera.position[0]);
	VectorCoord vecWorldup(m_camera.worldup, m_camera.worldup + sizeof m_camera.worldup / sizeof m_camera.worldup[0]);
	camDir = normalize(decreaseCoord(vecLookAt, vecPosition));
	camRight = normalize(cross(vecWorldup, camDir));
	camUp = normalize(decreaseCoord(vecWorldup, multiply(dot(vecWorldup, camDir), camDir)));

	float XiwData0[4][4] = {
		{camRight[X], camRight[Y], camRight[Z],0},
		{camUp[X], camUp[Y], camUp[Z], 0},
		{camDir[X], camDir[Y], camDir[Z],0},
		{0, 0, 0, 1}
	};
	float XiwData1[4][4] = {
	{1, 0, 0, -dot(camRight, vecPosition)},
	{0, 1, 0, -dot(camUp, vecPosition)},
	{0,0, 1, -dot(camDir, vecPosition)},
	{0, 0, 0, 1}
	};
	//VectorMatrix Xiw(4, std::vector<float>(4, 0));


	//for (int i = 0; i < 4; ++i) {
	//	for (int j = 0; j < 4; ++j) {
	//		Xiw[i][j] = XiwData[i][j];
	//	}
	//}

	float d = tan((m_camera.FOV / 2) * (PI / 180));
	float XpiData[4][4] = {
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, d, 0},
		{0, 0, d, 1}
	};
	//VectorMatrix Xpi(4, std::vector<float>(4, 0));
	//for (int i = 0; i < 4; ++i) {
	//	for (int j = 0; j < 4; ++j) {
	//		Xpi[i][j] = XpiData[i][j];
	//	}
	//}

	float XspData[4][4] = {
		{xres * 0.5, 0, 0, xres * 0.5},
		{0, -yres * 0.5, 0, yres * 0.5},
		{0, 0,  INT_MAX, 0},
		{0, 0, 0, 1}
	};
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			Xsp[i][j] = XspData[i][j];
		}
	}
	//VectorMatrix Xsp0(4, std::vector<float>(4, 0));
	//for (int i = 0; i < 4; ++i) {
	//	for (int j = 0; j < 4; ++j) {
	//		Xsp0[i][j] = XspData[i][j];
	//	}
	//}

	//VectorMatrix Xsw = multiplyMatrix(multiplyMatrix(Xsp0, Xpi), Xiw);

	matlevel = -1;
	//GzMatrix floatXsw = {
	//	{Xsw[0][0],Xsw[0][1],Xsw[0][2],Xsw[0][3]},
	//	{Xsw[1][0],Xsw[1][1],Xsw[1][2],Xsw[1][3]},
	//	{Xsw[2][0],Xsw[2][1],Xsw[2][2],Xsw[2][3]},
	//	{Xsw[3][0],Xsw[3][1],Xsw[3][2],Xsw[3][3]},
	//};
	GzPushMatrix(XspData);
	GzPushMatrix(XpiData);
	GzPushMatrix(XiwData1);
	GzPushMatrix(XiwData0);

	return GZ_SUCCESS;
}

int GzRender::GzPutCamera(GzCamera camera)
{
	/* HW 3.8
	/*- overwrite renderer camera structure with new camera definition
	*/
	m_camera = camera;
	return GZ_SUCCESS;
}

int GzRender::GzPushMatrix(GzMatrix matrix)
{
	/* HW 3.9

	push a matrix onto the Ximage stack

	check for stack overflow
	*/

	if (matlevel >= MATLEVELS) {
		// Stack Overflow
		return GZ_FAILURE;
	}
	++matlevel;
	bool isRotateMatrix = true;
	for (int i = 0; i < 4; i++) {
		float rowLengthSq = 0;
		float colLengthSq = 0;
		for (int j = 0; j < 4; j++) {
			rowLengthSq += matrix[i][j] * matrix[i][j];
			colLengthSq += matrix[j][i] * matrix[j][i];
		}
		if (abs(sqrt(rowLengthSq) - 1) > 0.0001 || abs(sqrt(colLengthSq) - 1) > 0.0001) {
			// The matrix is not a unitary rotate matrix
			isRotateMatrix = false;
		}
	}


	if (isRotateMatrix) {
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				Xnorm[matlevel][i][j] = matrix[i][j];
			}
		}
	}
	else {
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				if (i == j) {
					Xnorm[matlevel][i][j] = 1.0f;
				}
				else {
					Xnorm[matlevel][i][j] = 0.0f;
				}
			}
		}
	}
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			Ximage[matlevel][i][j] = matrix[i][j];
		}
	}

	return GZ_SUCCESS;
}

int GzRender::GzPopMatrix()
{
	/* HW 3.10

	pop a matrix off the Ximage stack

	check for stack underflow
	*/

	if (matlevel < 0) {
		// Stack Underflow
		return GZ_FAILURE;
	}

	matlevel--;
	return GZ_SUCCESS;
}

int GzRender::GzPut(int i, int j, GzIntensity r, GzIntensity g, GzIntensity b, GzIntensity a, GzDepth z)
{
	/* HW1.4 write pixel values into the buffer */
	if (i >= 0 && i < xres && j >= 0 && j < yres) {
		int index = ARRAY(i, j);



		if (z < pixelbuffer[index].z) {
			pixelbuffer[index].z = z;
			pixelbuffer[index].red = r;
			pixelbuffer[index].green = g;
			pixelbuffer[index].blue = b;
			pixelbuffer[index].alpha = a;
		}



		return GZ_SUCCESS;
	}
	else
	{
		return GZ_FAILURE;
	}
	//return GZ_SUCCESS;
}


int GzRender::GzGet(int i, int j, GzIntensity* r, GzIntensity* g, GzIntensity* b, GzIntensity* a, GzDepth* z)
{
	/* HW1.5 retrieve a pixel information from the pixel buffer */
	if (i >= 0 && i < xres && j >= 0 && j < yres) {
		int index = ARRAY(i, j);

		*r = pixelbuffer[index].red;
		*g = pixelbuffer[index].green;
		*b = pixelbuffer[index].blue;
		*a = pixelbuffer[index].alpha;
		*z = pixelbuffer[index].z;

		return GZ_SUCCESS;
	}
	else {
		return GZ_FAILURE;
	}
	return GZ_SUCCESS;
}



int GzRender::GzFlushDisplay2File(FILE* outfile)
{
	/* HW1.6 write image to ppm file -- "P6 %d %d 255\r" */
	fprintf(outfile, "P6 %d %d 255\n", xres, yres);
	for (int i = 0; i < xres * yres; ++i) {
		// store the high 8 bits

		fputc((pixelbuffer[i].red >> 4) > 255 ? 255 : pixelbuffer[i].red >> 4, outfile);

		fputc((pixelbuffer[i].green >> 4) > 255 ? 255 : pixelbuffer[i].green >> 4, outfile);

		fputc((pixelbuffer[i].blue >> 4) > 255 ? 255 : pixelbuffer[i].blue >> 4, outfile);

	}
	return GZ_SUCCESS;
}

int GzRender::GzFlushDisplay2FrameBuffer()
{
	/* HW1.7 write pixels to framebuffer:
		- put the pixels into the frame buffer
		- CAUTION: when storing the pixels into the frame buffer, the order is blue, green, and red
		- NOT red, green, and blue !!!
	*/
	for (int i = 0; i < xres * yres; ++i) {
		*(framebuffer + i * 3 + 0) = (char)((pixelbuffer[i].blue >> 4) > 255 ? 255 : pixelbuffer[i].blue >> 4);
		*(framebuffer + i * 3 + 1) = (char)((pixelbuffer[i].green >> 4) > 255 ? 255 : pixelbuffer[i].green >> 4);
		*(framebuffer + i * 3 + 2) = (char)((pixelbuffer[i].red >> 4) > 255 ? 255 : pixelbuffer[i].red >> 4);
	}
	return GZ_SUCCESS;
}


/***********************************************/
/* HW2 methods: implement from here */

int GzRender::GzPutAttribute(int numAttributes, GzToken* nameList, GzPointer* valueList) {
	for (int i = 0; i < numAttributes; i++) {
		switch (nameList[i]) {
			case GZ_RGB_COLOR: {
				GzColor* color = (GzColor*)valueList[i];
				this->flatcolor[RED] = (*color)[RED];
				this->flatcolor[GREEN] = (*color)[GREEN];
				this->flatcolor[BLUE] = (*color)[BLUE];
				break;
			}
			case GZ_INTERPOLATE: {
				int* mode = (int*)valueList[i];
				this->interp_mode = *mode;
				break;
			}
			case GZ_DIRECTIONAL_LIGHT: {
				if (this->numlights < MAX_LIGHTS) {
					GzLight* light = (GzLight*)valueList[i];
					this->lights[this->numlights] = *light;
					this->numlights++;
				}
				break;
			}
			case GZ_AMBIENT_LIGHT: {
				GzLight* ambient = (GzLight*)valueList[i];
				this->ambientlight = *ambient;
				break;
			}
			case GZ_AMBIENT_COEFFICIENT: {
				GzColor* ka = (GzColor*)valueList[i];
				Ka[RED] = (*ka)[RED];
				Ka[GREEN] = (*ka)[GREEN];
				Ka[BLUE] = (*ka)[BLUE];
				break;
			}
			case GZ_DIFFUSE_COEFFICIENT: {
				GzColor* kd = (GzColor*)valueList[i];
				Kd[RED] = (*kd)[RED];
				Kd[GREEN] = (*kd)[GREEN];
				Kd[BLUE] = (*kd)[BLUE];
				break;
			}
			case GZ_SPECULAR_COEFFICIENT: {
				GzColor* ks = (GzColor*)valueList[i];
				Ks[RED] = (*ks)[RED];
				Ks[GREEN] = (*ks)[GREEN];
				Ks[BLUE] = (*ks)[BLUE];
				break;
			}
			case GZ_DISTRIBUTION_COEFFICIENT: {
				float* sp = (float*)valueList[i];
				this->spec = *sp;
				break;
			}
			case GZ_TEXTURE_MAP: {
				GzTexture sp = (GzTexture)valueList[i];
				this->tex_fun = sp;
				break;
			}
		}
	}
	return GZ_SUCCESS;
}

float interpolate(float a, float b, float t) {
	return a + t * (b - a);
}

VectorCoord interpolateColor(GzColor v1, GzColor v2, float t)
{
	VectorCoord result(3);

	for (int i = 0; i < 3; i++)
	{
		result[i] = (1 - t) * v1[i] + t * v2[i];
	}

	return result;
}

VectorCoord interpolateColor(VectorCoord v1, VectorCoord v2, float t)
{
	VectorCoord result(3);

	for (int i = 0; i < 3; i++)
	{
		result[i] = (1 - t) * v1[i] + t * v2[i];
	}

	return result;
}

VectorCoord interpolateTextureColor(VectorCoord v1, VectorCoord v2, float t)
{
	VectorCoord result(2);

	for (int i = 0; i < 2; i++)
	{
		result[i] = (1 - t) * v1[i] + t * v2[i];
	}

	return result;
}

GzVertex getTriangleNormal(GzTriangle triangle, GzVertex intersection) {
    GzVertex A = triangle.v[0];
    GzVertex B = triangle.v[1];
    GzVertex C = triangle.v[2];

    double x = intersection.position[0];
    double y = intersection.position[1];
    double x0 = A.position[0], y0 = A.position[1];
    double x1 = B.position[0], y1 = B.position[1];
    double x2 = C.position[0], y2 = C.position[1];

    double alpha = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / ((y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2));
    double beta = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / ((y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2));
    double gamma = 1.0f - alpha - beta;
    GzVertex normal;

    for (int i = 0; i < 3; i++) {
        normal.normal[i] = alpha * A.normal[i] + beta * B.normal[i] + gamma * C.normal[i];
    }

    // Normalizing
    double length = sqrt(normal.normal[0] * normal.normal[0] + normal.normal[1] * normal.normal[1] + normal.normal[2] * normal.normal[2]);
    for (int i = 0; i < 3; i++) {
        normal.normal[i] /= length;
    }

    return normal;
}

/**
// if we don't use Phong model, then isInShadow is not needed because recursion can also produce shadow
bool GzRender::isInShadow(GzVertex intersection, GzLight light) {

	GzRay shadowRay;
	shadowRay.startPoint = intersection;
	shadowRay.direction.position[0] = light.position[0] - intersection.position[0];
	shadowRay.direction.position[1] = light.position[1] - intersection.position[1];
	shadowRay.direction.position[2] = light.position[2] - intersection.position[2];

	
	for (int i = 0; i < numTriangles; i++) {
		int t; // 
		//
		bool collision = GzCollisionWithSpecificTriangle(shadowRay, triangles[i], t);

		// 
		// 
		if (collision && t >= 0 && t <= 1) {
			return true;
		}
	}

	// 
	return false;
}
*/

void GzRender::RayTrace(){
	float d = tan((m_camera.FOV / 2) * (PI / 180));
	float aspect_ratio = 1.0 * xres / yres;
	for(int i = 0; i < xres; i++) {
		for(int j = 0; j < yres; j++) {
			float x = (2 * (i + 0.5) / xres - 1) * d * aspect_ratio;
			float y = (1 - 2 * (j + 0.5) / yres) * d;
			GzVector3D color = EmitLight(GzRay(GzVector3D(0, 0, -1), GzVector3D(x, y, 1)), 1);

			GzDepth z = 0;
			GzPut(i, j, ctoi(color[RED]), ctoi(color[GREEN]), ctoi(color[BLUE]), 1, z);
		}
	}
		
}

/**
 * Given a light and a triangle(need to be intersected beforehand), get its reflective and refractive light
 *
 * @param light The input light(The input direction)
 * @param intersection The intersection point of the light and the triangle
 * @param triangle The input triangle(collision)
 * @param depth Recursive depth
 * @return The total color(from reflect, refract, etc) generated by this Fresnel reflection
 */
GzVector3D GzRender::FresnelReflection(GzRay light, GzVertex intersection, GzTriangle triangle, int depth)
{
	// Equation source: An improved illumination model for shaded display, Sec 2 Improved model
	// The total light color will be: (ambient color) + kd * (sum of diffuse lights) + ks * (reflect light, also known as specular) + kt * (refract light)

	GzVertex N_vertex = getTriangleNormal(triangle, intersection);	// Compute normal vector at intersection point
	GzVector3D normal = GzVector3D(N_vertex.normal);
	GzVector3D intersection_pos = GzVector3D(intersection.position);
	float n_air = 1; // Refractive index of air, 1.0003
	float n1, n2;	// (Incoming n1), (refracting n2) for the refractive index

	// If dot of normal and light are positive, then the light comes from air(same direction)
	GzVector3D lightDir = GzVector3D(light.direction);

	if (lightDir * normal > 0)
	{
		// Then air will be incoming n1, the material will be refract n2
		n1 = n_air;
		n2 = triangle.v[0].refract_index;
	}
	else
	{
		// Then material will be incoming n, the air will be refract n
		n1 = triangle.v[0].refract_index;
		n2 = n_air;
	}

	// Get cos & sin of incoming light vector and normal vector
	float cos_incoming = lightDir * normal;
	float sin_incoming = sqrt(1 - pow(cos_incoming, 2));	// sin2 + cos2 = 1

	// Compute Directions
	GzVector3D reflectDir = lightDir - 2 * normal * cos_incoming;	// Reflect = L - 2(L dot N)N
	float sin_refracting = (n1 / n2) * sin_incoming;	// n1 * sin_incoming = n2 * sin_refracting
	float cos_refracting = sqrt(1 - pow(sin_refracting, 2));	// sin2 + cos2 = 1
	// Calculate the incoming light's vector portion that is perpendicular to normal, which call tangent vector
	GzVector3D tangentVec = (lightDir - cos_incoming * normal) / sin_incoming;	// Tangent vector portion of incoming light
	// Convert tangent vector into refract light direction
	GzVector3D refractDir = tangentVec * sin_refracting - normal * cos_refracting;


	// Schlick’s model
	float r0 = pow(n1 - n2, 2) / pow(n1 + n2, 2);

	// Get Reflect and Refract ratio
	float reflectRatio = r0 + (1 - r0) * (1 - cos_incoming);
	float refractRatio = 1 - reflectRatio;


	// Compute Rays
	GzRay reflectedRay = GzRay(intersection_pos, reflectDir);
	GzRay refractedRay = GzRay(intersection_pos, refractDir);

	// Calculate the color of the reflection
	GzVector3D reflectedColor = EmitLight(reflectedRay, depth + 1);
	GzVector3D refractedColor = EmitLight(refractedRay, depth + 1);

	// Compute total color
	// TODO: Diffuse color formula undecided
	float r1 = (rand() / (float)RAND_MAX) * 2 * PI;	// Rand samples fron 0 to RAND_MAX, scale it to 2 pi
	float r2 = (rand() / (float)RAND_MAX) * PI / 2;	// vertical angle from 0 to pi / 2

	// N is normal to the current point, we need to find the x y plane perpendicular to it
	GzVector3D NX = GzVector3D(1, 1, -(normal[0] + normal[1]) / normal[2]).normalized();
	GzVector3D NY = (normal ^ NX).normalized();

	GzVector3D diffuseDir = (NX * cos(r1) * sin(r2) + NY * sin(r1) * sin(r2) + normal * cos(r2)).normalized();
	GzRay diffuseRay = GzRay(intersection_pos, diffuseDir);

	// Color of incoming light for diffuse
	GzVector3D diffuseRayColor = EmitLight(diffuseRay, depth + 1);
	// Color of outgoing light for diffuse, le * (N dot L)
	// Total color
	GzVector3D diffuseColor = /**Kd * */diffuseRayColor * (normal * diffuseDir);
	GzVector3D totalColor = reflectRatio * reflectedColor + refractRatio * refractedColor + diffuseColor;

	return totalColor;
}

GzVector3D GzRender::EmitLight(GzRay ray, int depth) 
{
	int maxDepth = 5;
	GzVector3D normDirection = GzVector3D(ray.direction).normalized();
	// Check the intersection between the light beam and objects in the scene
	GzVertex intersection; // intersection point between ray and object?
	GzTriangle collidedTriangle; // The intersecting triangle(collision)
    //Ray intersection;
	// intersect either sphere or triangle
	int index;

	// TODO: negotiate how to call intersectScene (Kevin)
    if (!GzCollisionWithTriangle(ray, index)/*intersectObject(ray, intersection)*/) 
	{ 
        return GzVector3D(0, 0, 0); // just black is ok
    }

	//GzVertex normal = getTriangleNormal(*intersectedTriangle, );

	// Calculate the color based on the Phong model
    //VectorCoord localColor = phongModel(Ray(startPoint, direction), hit);
	GzVector3D localColor = FresnelReflection(ray, intersection, triangles[index], depth);

	// If the set maximum recursive depth is reached, no further reflection computation occurs
	if (depth >= maxDepth)
	{
		return GzVector3D(0, 0, 0);
	}
    // Light source for the final color.
    // 1. Define the radius of the light source.
    // 2. Do the collision detection by CollisionWithSphere.
    // 3. Set the light source color.    One white light and one color light.
	
	for (GzSphere sp : lightSources)
	{
		double dist_placeholder;
		if (RayIntersectsSphere(ray.startPoint.GetDoubleArr(), ray.direction.GetDoubleArr(), sp.position, sp.radius, dist_placeholder))
		{
			return GzVector3D(sp.color_diffuse);
		}
	}
    // The overall color is a combination of the color computed from the Phong model and the color of the reflection
    //VectorCoord color = localColor + reflectedColor * 0.8;
    
    return localColor;
}

/**
 * Given a light, return if the light is colliding with any triangle
 *
 * @param light The input light for collision detection
 * @param index The output index of first colliding triangle
 * @return A boolean indiciating if the light is colliding with any triangle
 */
bool GzRender::GzCollisionWithTriangle(GzRay light, int& index)
{
	index = -1;
	GzVector3D firstIntersectPos;
	for (int i = 0; i < triangleNum; i++)
	{
		GzVector3D currIntersectPos;
		if (GzCollisionWithSpecificTriangle(light, triangles[i], currIntersectPos))
		{
			// If intersects, check if other triangle has collided with the light yet
			if (index == -1)
			{
				// Light not collide with other triangles yet
				index = i;
				firstIntersectPos = currIntersectPos;
			}
			else
			{
				// Light collided with other triangles before, check intersection position

				// Intersection points are points on light direction vector, so we only need to compare one of the dimensions.
				// We need to select a non-zero value from one of the dimensions.
				int j;
				for (j = 0; j < 3; j++)
				{
					if (light.direction[j] > 0) break;
				}

				// If the "firstPos-->currPos" vector and the light direction vector are having different signs, 
				// then currPos will be the first point the light is colliding.
				double diff = currIntersectPos[j] - firstIntersectPos[j];
				if (diff / light.direction[0] < 0)
				{
					// Then current triangle will be the first triangle intersecting
					index = i;
					firstIntersectPos[0] = currIntersectPos[0];
					firstIntersectPos[1] = currIntersectPos[1];
					firstIntersectPos[2] = currIntersectPos[2];
				}
			}
		}
	}
	// If index is valid return true
	return index != -1;
}

/**
 * Given a light, return if the light is colliding with a specific triangle
 *
 * @param light The input light for collision detection
 * @param triangle The input triangle for collision detection
 * @param intersectPos The intersecting point as array pointer
 * @return A boolean indiciating if the light is colliding with the input triangle
 */
bool GzRender::GzCollisionWithSpecificTriangle(GzRay light, GzTriangle triangle, GzVector3D intersectPos)
{
	GzVector3D vertexA_pos = GzVector3D(triangle.v[0].position);
	GzVector3D vertexB_pos = GzVector3D(triangle.v[1].position);
	GzVector3D vertexC_pos = GzVector3D(triangle.v[2].position);
	// Cross product of two edges that normal to triangle-position-plane, normalized
	GzVector3D triangle_plane_norm = ((vertexB_pos - vertexA_pos) ^ (vertexC_pos - vertexA_pos)).normalized();	

	// If flat triangle normal perpendicular to light, then no intersection in terms of direction
	if (abs(triangle_plane_norm * light.direction) < 0.0001)
	{
		return false;
	}

	// Compute intersection point(with triangle's plane)
	// d = plane_normal dot x
	// t = (d - plane_normal dot origin) / (plane_normal dot direction)
	float d = triangle_plane_norm * vertexA_pos;
	double t = (d - (triangle_plane_norm * light.startPoint)) / (triangle_plane_norm * light.direction);
	intersectPos = light.startPoint + t * light.direction;

	// Check if the point is within triangle
	GzVector3D AB = vertexB_pos - vertexA_pos;
	GzVector3D AQ = intersectPos - vertexA_pos;
	bool test1 = ((AB ^ AQ) * triangle_plane_norm) >= 0;

	GzVector3D BC = vertexC_pos - vertexB_pos;
	GzVector3D BQ = intersectPos - vertexB_pos;
	bool test2 = ((BC ^ BQ) * triangle_plane_norm) >= 0;
	
	GzVector3D CA = vertexA_pos - vertexC_pos;
	GzVector3D CQ = intersectPos - vertexC_pos;
	bool test3 = ((CA ^ CQ) * triangle_plane_norm) >= 0;

	return test1 && test2 && test3;
}

int GzRender::GzPutTriangle(int numParts, GzToken* nameList, GzPointer* valueList)
/* numParts - how many names and values */
{
	/* HW 2.2
	-- Pass in a triangle description with tokens and values corresponding to
		  GZ_NULL_TOKEN:		do nothing - no values
		  GZ_POSITION:		3 vert positions in model space
	-- Invoke the rastrizer/scanline framework
	-- Return error code
	*/
	GzCoord* vertices = NULL;
	GzCoord* normals = NULL;
	GzTextureIndex* uvlist = NULL;
	for (int i = 0; i < numParts; i++) {
		if (nameList[i] == GZ_POSITION) {
			vertices = static_cast<GzCoord*>(valueList[i]);
		}
		else if (nameList[i] == GZ_NORMAL) {
				normals = static_cast<GzCoord*>(valueList[i]);
			}
		else if (nameList[i] == GZ_TEXTURE_INDEX) {
			uvlist = static_cast<GzTextureIndex*>(valueList[i]);
		}
	}

	if (vertices != NULL) {
		// Sort vertices by y
		std::vector<std::vector<float>> verticesVec, normalVec, UVlistVec;
		std::vector<std::vector<float>> outputVec(3, std::vector<float>(3));
		// Add each GzCoord to the vector
		for (int i = 0; i < 3; ++i)
		{
			verticesVec.push_back(std::vector<float>(vertices[i], vertices[i] + 3));
			normalVec.push_back(std::vector<float>(normals[i], normals[i] + 3));
			UVlistVec.push_back(std::vector<float>(uvlist[i], uvlist[i] + 2));
		}
		for (int count = matlevel; count >= 2; count--) {
			GzMatrix& matrix = Ximage[count];
			GzMatrix& Xn = Xnorm[count];
			for (int h = 0; h < 3; ++h) {
				std::vector<float> resultVec0(4);  // ������� ��ʼֵΪ0
				std::vector<float> resultVec1(4);  // ������� ��ʼֵΪ0
				verticesVec[h].push_back(1.0);  // ��չ��4D����
				normalVec[h].push_back(1.0);
				for (int j = 0; j < 4; ++j) {
					for (int k = 0; k < 4; ++k) {
						resultVec0[j] += matrix[j][k] * verticesVec[h][k];
						resultVec1[j] += Xn[j][k] * normalVec[h][k];
					}
				}
				verticesVec[h].pop_back();
				normalVec[h].pop_back();
				verticesVec[h][0] = resultVec0[0] / resultVec0[3];
				verticesVec[h][1] = resultVec0[1] / resultVec0[3];
				verticesVec[h][2] = resultVec0[2] / resultVec0[3];
				normalVec[h][0] = resultVec1[0] / resultVec1[3];
				normalVec[h][1] = resultVec1[1] / resultVec1[3];
				normalVec[h][2] = resultVec1[2] / resultVec1[3];
				if (count == 2 && verticesVec[h][2] < 0) {
					return GZ_FAILURE;
				}
			}
		}
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				vertices[i][j] = verticesVec[i][j];
				normals[i][j] = normalVec[i][j];

			}
		}
		if (vertices != nullptr && normals != nullptr && numTriangles < MAX_TRIANGLES)
		{
			for (int i = 0; i < 3; i++)
			{
				GzVertex v(vertices[i], normals[i]);
				triangles[numTriangles] = GzTriangle(v, v, v);
				numTriangles++;
			}
		}
	}
	return GZ_SUCCESS;
}

// Dot product for vector
template<typename T1, typename T2>
double dotArray(T1 a[3], T2 b[3]) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Cross product for vector
template<typename T1, typename T2, typename T3>
void crossArray(T1 a[3], T2 b[3], T3 c[3]) {
	c[0] = a[1] * b[2] - a[2] * b[1];
	c[1] = a[2] * b[0] - a[0] * b[2];
	c[2] = a[0] * b[1] - a[1] * b[0];
}

/**
 * Check if a ray intersects with a sphere.
 *
 * @param origin The origin of the ray
 * @param direction The direction of the ray
 * @param center The center of the sphere
 * @param radius The radius of the sphere
 * @param t A reference to store the distance from the origin to the intersection point
 * @return A boolean indicating if the ray intersects with the sphere
 */
bool GzRender::RayIntersectsSphere(const double origin[3], const double direction[3], const double center[3], double radius, double &t) 
{
    double oc[3] = { origin[0] - center[0], origin[1] - center[1], origin[2] - center[2] };
    double b = 2 * dotArray(oc, direction);
    double c = dotArray(oc, oc) - radius * radius;
    double discriminant = b * b - 4 * c;
    if (discriminant < 0) {
        return false; // No intersection
    } else {
        discriminant = sqrt(discriminant);
        double t0 = (-b - discriminant) / 2;
        double t1 = (-b + discriminant) / 2;
        t = (t0 < t1) ? t0 : t1;
        return true; // Intersection occurs
    }
}

/**
 * Given a light, return if the light is colliding with any sphere in the scene.
 *
 * @param light The input light for collision detection
 * @param index The output index of first colliding sphere
 * @return A boolean indicating if the light is colliding with any sphere
 */
bool GzRender::GzCollisionWithSphere(GzRay light, int &index) 
{
    index = -1;
	
    double closest_t = DBL_MAX; //std::numeric_limits<double>::max(); identifier error
    double t;
    for (int i = 0; i < sphereNum; i++) {
        if (RayIntersectsSphere(light.startPoint.GetDoubleArr(), light.direction.GetDoubleArr(), spheres[i].position, spheres[i].radius, t)) {
            if (t < closest_t) {
                closest_t = t;
                index = i;
            }
        }
    }
    return index != -1;
}

/**
 * Calculates the reflection and refraction vectors given an intersection point with a sphere.
 *
 * @param light The incoming light
 * @param sphere The sphere with which the light interacts
 * @param intersectionPoint The point of intersection
 * @param reflectLight The reflected light
 * @param refractLight The refracted light
 */
void GzRender::CalculateSphereReflectionAndRefraction(GzRay light, GzSphere sphere, const double intersectionPoint[3], GzRay&reflectLight, GzRay&refractLight)
{
    // Calculate normal at the intersection point
    double normal[3] = {
        (intersectionPoint[0] - sphere.position[0]) / sphere.radius,
        (intersectionPoint[1] - sphere.position[1]) / sphere.radius,
        (intersectionPoint[2] - sphere.position[2]) / sphere.radius
    };

    // Calculate reflection vector
    double dot_ln = dotArray(light.direction.GetDoubleArr(), normal);
    reflectLight.direction[0] = light.direction[0] - 2 * dot_ln * normal[0];
    reflectLight.direction[1] = light.direction[1] - 2 * dot_ln * normal[1];
    reflectLight.direction[2] = light.direction[2] - 2 * dot_ln * normal[2];

    // TODO: refraction
}