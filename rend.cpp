
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
#include <Windows.h>
#include <string>

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
	numTriangles = 0;
	numlights = 0;
	sphereNum;
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
GzVector3D ClampVector(GzVector3D vec) {
	for (int i = 0; i < 3; ++i) {
		vec.arr[i] = max(0.0f, min(1.0f, vec.arr[i]));
	}
	return vec;
}
void GzRender::RayTrace() {
	float d = tan((m_camera.FOV / 2) * (PI / 180));
	float aspect_ratio = 1.0 * xres / yres;
	for (int i = 0; i < xres; i++) {
		for (int j = 0; j < yres; j++) {
			float x = (2 * (i + 0.5) / xres - 1) * d * aspect_ratio;
			float y = (1 - 2 * (j + 0.5) / yres) * d;
			GzVector3D color = ClampVector(EmitLight(GzRay(GzVector3D(0, 0, -1), GzVector3D(x, y, 1)), 1,1));
			GzDepth z = 0;
			GzPut(i, j, ctoi(color[RED]), ctoi(color[GREEN]), ctoi(color[BLUE]), 1, z);
		}
		std::string info = "Rendering: (" + std::to_string(i + 1) + "/" + std::to_string(xres) + ")\n";
		OutputDebugStringA(info.c_str());
	}
}


GzVector3D GzRender::EmitLight(GzRay ray, int depth,int reflected_depth) 
{
	int maxDepth =6;
	// If the set maximum recursive depth is reached, no further reflection computation occurs
	if (depth >= maxDepth)
	{
		return GzVector3D(0, 0, 0);
	}

	GzVector3D intersection; // intersection point between ray and object?
	int index;
	// TODO: negotiate how to call intersectScene (Kevin)
    if (!GzCollisionWithTriangle(ray, index, intersection)/*intersectObject(ray, intersection)*/)
	{ 
        return GzVector3D(0, 0, 0); // just black is ok
    }
	GzVertex intersection_pos = GzVertex(intersection[0], intersection[1], intersection[2]);
	GzVertex intersection_norm = getTriangleNormal(triangles[index], intersection_pos);
	GzVertex intersection_vertex = GzVertex(intersection_pos.position, intersection_norm.normal);
	
	GzRay lightSource = GzRay(GzVector3D(0, 100, -1), GzVector3D(0, -1, 1), GzVector3D(1, 1, 1));
	// Initialize color with ambient light
	GzVector3D ambient = GzVector3D(Ka) & lightSource.color;

	return BSDFModel(ray, intersection_vertex, lightSource, index, depth, reflected_depth);
}

double clamp(double value, double low, double high)
{
	return max(min(value, 1.0), 0.0);
}

GzVector3D GzRender::PhongModel(GzRay ray, GzVertex intersection, GzRay lightSource, int triangleIndex, int depth,int reflected_depth) {
	// Acquire light position, object position and viewpoint position.
	GzVector3D light_pos = lightSource.startPoint;
	GzVector3D obj_pos = GzVector3D(intersection.position);
	GzVector3D obj_norm = GzVector3D(intersection.normal);
	GzVector3D view_pos = ray.startPoint;

	// The vector from the surface to light source (normalized) and the vector from the surface to viewer (normalized)
	GzVector3D light_vec = ( light_pos- obj_pos).normalized();
	GzVector3D view_vec = (view_pos- obj_pos).normalized();
	
	// Calculate L (light direction), N (normal), R (reflected light direction), and View (view direction)
	GzVector3D L = light_vec.normalized();
	GzVector3D N = obj_norm.normalized();
	
	//GzVector3D View = (obj_pos-GzVector3D(ray.startPoint) ).normalized();
	float dotProductNormLight = N* light_vec;
	float dotProductNormView = N* view_vec;
	if (dotProductNormLight < 0 && dotProductNormView < 0) {
		// flip normal vector if dot product under zero
		N[0] = -N[0];
		N[1] = -N[1];
		N[2] = -N[2];
		dotProductNormLight = -dotProductNormLight;
		dotProductNormView = -dotProductNormView;
	}
	GzVector3D R = (2.0 * (N * L) * N - L).normalized();
	// Calculate Phong shading components: diffuse and specular
	//double diffuse = clamp(L * N, 0.0, 1.0);
	//double specular = clamp(pow(clamp(R * View, 0.0, 1.0), spec), 0.0, 1.0);

	GzVector3D triangleColor = GzVector3D(triangles[triangleIndex].v[0].color);
	//GzVector3D triangleColor = GzVector3D(1, 1, 0.5);
	GzVector3D kd = triangleColor;
	GzVector3D ks = triangleColor;

	// Compute the diffuse part
	GzVector3D diffuse = kd & lightSource.color * dotProductNormLight;

	// Compute specular part
	float viewerDotReflection = R * view_vec;
	float PowReflection = pow(clamp(viewerDotReflection, 0.0, 1.0), 2);
	GzVector3D specular = ks & lightSource.color * PowReflection;


	GzVector3D firstIntersectPos;
	int index;

	GzRay shadowLight = GzRay(obj_pos, light_vec);
	if (GzCollisionWithTriangle(shadowLight, index, firstIntersectPos, triangleIndex))
	{
		diffuse = GzVector3D(0, 0, 0);
		specular = GzVector3D(0, 0, 0);
	}
	if (dotProductNormLight * dotProductNormView < 0) {
		diffuse = GzVector3D(0, 0, 0);
		specular = GzVector3D(0, 0, 0);
	}

	GzRay reflect_ray = GzRay(obj_pos, R); 
	float reflectance = 0.5;
	GzVector3D reflectColor = EmitLight(reflect_ray, depth + 1,reflected_depth+1);
	
	if (reflected_depth > 1) {
		reflectColor = GzVector3D(0, 0, 0);
	}
	GzVector3D color = (1.0f - reflectance) * (reflectColor) + reflectance * ( diffuse +  specular) ;
	return color;
}
float FresnelApproximation(float cosTheta, float reflectionIndex) {
	float r0 = (1 - reflectionIndex) / (1 + reflectionIndex);
	r0 = r0 * r0;
	return r0 + (1 - r0) * pow(1 - cosTheta, 5);
}

GzVector3D GzRender::BSDFModel(GzRay ray, GzVertex intersection, GzRay lightSource, int triangleIndex, int depth,int reflected_depth) {
	// Acquire light position, object position and viewpoint position.
	GzVector3D light_pos = lightSource.startPoint;
	GzVector3D obj_pos = GzVector3D(intersection.position);
	GzVector3D obj_norm = GzVector3D(intersection.normal);
	GzVector3D view_pos = ray.startPoint;

	// The vector from the surface to light source (normalized) and the vector from the surface to viewer (normalized)
	GzVector3D light_vec = (light_pos - obj_pos).normalized();
	GzVector3D view_vec = (view_pos - obj_pos).normalized();

	// Calculate L (light direction), N (normal), R (reflected light direction), and View (view direction)
	GzVector3D L = light_vec.normalized();
	GzVector3D N = obj_norm.normalized();

	//GzVector3D View = (obj_pos-GzVector3D(ray.startPoint) ).normalized();
	float dotProductNormLight = N * light_vec;
	float dotProductNormView = N * view_vec;
	if (dotProductNormLight < 0 && dotProductNormView < 0) {
		// flip normal vector if dot product under zero
		N[0] = -N[0];
		N[1] = -N[1];
		N[2] = -N[2];
		dotProductNormLight = -dotProductNormLight;
		dotProductNormView = -dotProductNormView;
	}
	GzVector3D R = (2.0 * (N * L) * N - L).normalized();

	// Compute local color
	GzVector3D triangleColor = GzVector3D(triangles[triangleIndex].v[0].color);
	GzVector3D kd = triangleColor;
	GzVector3D ks = triangleColor;



	// Compute the diffuse part
	GzVector3D diffuse = kd & lightSource.color * clamp(dotProductNormLight, 0.0, 1.0);
	// Compute specular part
	float viewerDotReflection = R * view_vec;
	float PowReflection = pow(clamp(viewerDotReflection, 0.0, 1.0), 32);
	GzVector3D specular = lightSource.color * PowReflection;


	GzVector3D firstIntersectPos;
	int index;
	GzRay shadowLight = GzRay(obj_pos, light_vec);
	if (dotProductNormLight * dotProductNormView < 0) {
		diffuse = GzVector3D(0, 0, 0);
		specular = GzVector3D(0, 0, 0);
	}
	if (GzCollisionWithTriangle(shadowLight, index, firstIntersectPos, triangleIndex))
	{
		diffuse = GzVector3D(0, 0, 0);
		specular = GzVector3D(0, 0, 0);
		
	}
	GzRay reflect_ray = GzRay(obj_pos, R);


	float refractionIndex =triangles[triangleIndex].v[0].refract_index;
	float cos_theta_i = -(N * ray.direction);
	float fresnel = FresnelApproximation(cos_theta_i, refractionIndex); // Fresnel approximation
	GzVector3D localColor = fresnel * (diffuse + specular);
	float total_inner_reflection = 1.0 - refractionIndex * refractionIndex * (1.0 - cos_theta_i * cos_theta_i);
	
	GzVector3D reflection_color = fresnel * EmitLight(reflect_ray, depth + 1, reflected_depth+1);

	// Refraction Effect
	//GzVector3D refracted_dir = refractionIndex * ray.direction + (refractionIndex * cos_theta_i - sqrt(total_inner_reflection)) * N;
	//GzRay refracted_ray(obj_pos, refracted_dir, ray.color);
	//
	GzVector3D refracted_color = (1.0f - fresnel) * EmitLight(GzRay(obj_pos, ray.direction), depth + 1,reflected_depth);
	if (total_inner_reflection < 0) {
		refracted_color = GzVector3D(0, 0, 0);
	}
	if (reflected_depth >1 ) {
		reflection_color = GzVector3D(0, 0, 0);
	}
	return localColor + reflection_color+ refracted_color;
}

/**
 * Given a light, return if the light is colliding with any triangle
 *
 * @param light The input light for collision detection
 * @param index The output index of first colliding triangle
 * @param firstIntersectPos The first position of intersection
 * @param currentIndex The index of current using triangle(will not collide)
 * @return A boolean indiciating if the light is colliding with any triangle
 */
bool GzRender::GzCollisionWithTriangle(GzRay light, int& index, GzVector3D& firstIntersectPos, int currentIndex)
{
	index = -1;
	float t_min = 1000000;
	for (int i = 0; i < numTriangles; i++)
	{
		GzVector3D currIntersectPos;
		float t = -1;
		if (GzCollisionWithSpecificTriangle(light, triangles[i], currIntersectPos,t))
		{
			// If intersects, check if other triangle has collided with the light yet
			if (index == -1)
			{
				// Light not collide with other triangles yet
				index = i;
				firstIntersectPos = currIntersectPos;
				t_min = t;
			}
			else
			{
				// Light collided with other triangles before, check intersection position
				// The closet position to start point will be the first intersection

				

				if ((t < t_min))
				{
					index = i;
					t_min = t;
					firstIntersectPos = currIntersectPos;
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
bool GzRender::GzCollisionWithSpecificTriangle(GzRay ray, GzTriangle triangle, GzVector3D& intersectPos,float &t_min)
{
	GzVector3D e1, e2;
	GzVector3D P, Q, T;
	float det, inv_det, u, v;
	float t;
	GzVector3D vertexA_pos = GzVector3D(triangle.v[0].position);
	GzVector3D vertexB_pos = GzVector3D(triangle.v[1].position);
	GzVector3D vertexC_pos = GzVector3D(triangle.v[2].position);
	e1 = vertexB_pos - vertexA_pos;
	e2 = vertexC_pos - vertexA_pos;

	P = ray.direction ^ e2;

	det = e1 * P;

	if (det > -0.0001 && det < 0.0001)
		return false;

	inv_det = 1.0 / det;
	T = ray.startPoint - triangle.v[0].position;

	u = (T * P) * inv_det;

	if (u < 0.0 || u > 1.0)
		return false;

	Q = T ^ e1;
	v = (ray.direction * Q) * inv_det;

	if (v < 0.0 || u + v  > 1.0)
		return false;

	t = (e2 * Q) * inv_det;

	if (t > 0.0001)
	{
		intersectPos = ray.startPoint + ray.direction * t;
		t_min = t;
		return true;
	}

	return false;
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
	float* refractiveList = NULL;
	GzCoord* colorList = NULL;
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
		else if (nameList[i] == GZ_REFRACT_INDEX) {
			refractiveList = static_cast<float*>(valueList[i]);
		}
		else if (nameList[i] == GZ_TRIANGLE_COLOR) {
			colorList = static_cast<GzCoord*>(valueList[i]);
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
			GzVertex v0(vertices[0], normals[0]);
			v0.refract_index = refractiveList[0];
			v0.setColor(colorList[0]);

			GzVertex v1(vertices[1], normals[1]);
			v1.refract_index = refractiveList[1];
			v1.setColor(colorList[1]);

			GzVertex v2(vertices[2], normals[2]);
			v2.refract_index = refractiveList[2];
			v2.setColor(colorList[2]);

			triangles[numTriangles] = GzTriangle(v0, v1, v2);
			numTriangles++;
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