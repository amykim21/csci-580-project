
/* CS580 Homework 4 */

#include	"stdafx.h"
#include	"stdio.h"
#include	"math.h"
#include	"Gz.h"
#include	"rend.h"

#include <algorithm>
#include <vector>
#include <cmath>

#define PI (float) 3.14159265358979323846
#define DEG2RAD(degree) ((degree) * (PI / 180.0))
typedef std::vector<float> VectorCoord;
typedef std::vector<std::vector<float>> VectorMatrix;

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
// dot�������ڼ������������ĵ��
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

// cross�������ڼ������������Ĳ��
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
// �������
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

	// ����Camera������

	VectorCoord vecLookAt(m_camera.lookat, m_camera.lookat + sizeof m_camera.lookat / sizeof m_camera.lookat[0]);
	VectorCoord vecPosition(m_camera.position, m_camera.position + sizeof m_camera.position / sizeof m_camera.position[0]);
	VectorCoord vecWorldup(m_camera.worldup, m_camera.worldup + sizeof m_camera.worldup / sizeof m_camera.worldup[0]);
	camDir = normalize(decreaseCoord(vecLookAt, vecPosition));
	camRight = normalize(cross(vecWorldup, camDir));
	camUp = normalize(decreaseCoord(vecWorldup, multiply(dot(vecWorldup, camDir), camDir)));

	// ���� Xiw
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

	// ���� Xpi
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
	// ���� Xsp
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
	// ���� Xsw
	//VectorMatrix Xsw = multiplyMatrix(multiplyMatrix(Xsp0, Xpi), Xiw);

	// ���ۺ�ʱ Xsw ������ stack �ĵײ�
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
		}							// �ڴ˴����������Ӹ����������ô���
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

bool GzRender::isInShadow(GzVertex intersection, GzLight light) {

	GzRay shadowRay;
	shadowRay.startPoint = intersection;
	shadowRay.direction.position[0] = light.position[0] - intersection.position[0];
	shadowRay.direction.position[1] = light.position[1] - intersection.position[1];
	shadowRay.direction.position[2] = light.position[2] - intersection.position[2];

	
	for (int i = 0; i < numTriangles; i++) {
		double t; // 
		//
		bool collision = collisionWithTriangle(shadowRay, triangles[i], &t);

		// 
		// 
		if (collision && t >= 0 && t <= 1) {
			return true;
		}
	}

	// 
	return false;
}

VectorCoord getBackgroundColor(Ray ray) {
	// TODO
}
void GzRender:: RayTrace(){
	float d = tan((m_camera.FOV / 2) * (PI / 180));
	float aspect_ratio= 1.0*xRes/yRes
	for(int i =0;i<Xres;i++){
		for(int j =0;j<Yres;j++){
			x = (2 * (i + 0.5) / xRes - 1) * d * aspect_ratio;
				
			y = (1 - 2 * (j + 0.5) / yRes) * d;
			VectorCoord color=emitLight(Ray(GzVertex(0,0,-1),GzVertex(x,y,1)));
			GzPut(i, j, ctoi(color[RED]), ctoi(color[GREEN]), ctoi(color[BLUE]), 1, z);
		}
	}
		
}
VectorCoord phongModel(Ray ray, Intersection intersection, Light light){
    // Acquire light position, object position and viewpoint position.
    VectorCoord light_pos = light.position;
    VectorCoord obj_pos = intersection.position;
    VectorCoord view_pos = ray.origin;

    // The vector from the surface to light source (normalized) and the vector from the surface to viewer (normalized)
    VectorCoord light_vec = normalize(light_pos - obj_pos);
    VectorCoord view_vec = normalize(view_pos - obj_pos);

    // Compute reflection vector
    VectorCoord reflect_vec = reflect(negate(light_vec), intersection.normal);
   // Calculate L (light direction), N (normal), R (reflected light direction), and V (view direction)
    double L[3] = { light.position[0] - intersection.position[0],
                    light.position[1] - intersection.position[1],
                    light.position[2] - intersection.position[2] };
    normalize(L);

    double N[3] = { intersection.normal[0],
                    intersection.normal[1],
                    intersection.normal[2] };
    normalize(N);

    double R[3] = { 2.0 * dot_product(N, L) * N[0] - L[0],
                    2.0 * dot_product(N, L) * N[1] - L[1],
                    2.0 * dot_product(N, L) * N[2] - L[2] };
    normalize(R);

    double V[3] = { camera_pos[0] - intersection.position[0],
                    camera_pos[1] - intersection.position[1],
                    camera_pos[2] - intersection.position[2] };
    normalize(V);

    // Calculate Phong shading components: diffuse and specular
    double diffuse = clamp(dot_product(L, N), 0.0, 1.0);
    double specular = clamp(pow(clamp(dot_product(R, V), 0.0, 1.0), vertex.shininess), 0.0, 1.0);
    // Initialize color with ambient light
    VectorCoord color = Ka * light.color;

    // Compute the diffuse part
    VectorCoord diff = Kd * light.color * std::max(dotProduct(intersection.normal, light_vec), 0.0);

    // Compute specular part
    VectorCoord spec = Ks * light.color * pow(dotProduct(reflect_vec, view_vec), spec);

    // If the object is in shadow, set the diffuse and specular part to 0
    if (isInShadow(intersection, light)) {
        diff = VectorCoord(0, 0, 0);
        spec = VectorCoord(0, 0, 0);
    }

    // Total color is the sum of the ambient, diffuse and specular color
    color += (diff + spec);

    return color;
}

VectorCoord emitLight(Ray ray, int depth) {
	int maxDepth = 5;
    VectorCoord normDirection = normalize(direction);
	// Check the intersection between the light beam and objects in the scene
    VectorCoord hit;
    Ray intersection;
    if (!intersectScene(Ray(startPoint, direction), hit, intersection)) {
        return getBackgroundColor(Ray(startPoint, direction));
    }

	GzVertex normal = getTriangleNormal(*intersectedTriangle, );

	// Calculate the color based on the Phong model
    VectorCoord localColor = phongModel(Ray(startPoint, direction), hit);

	// If the set maximum recursive depth is reached, no further reflection computation occurs
    if (depth >= maxDepth)
        return localColor;
    
    VectorCoord R = reflect(direction, hit);
    Ray reflectedRay(hit, R);
    
    // Calculate the color of the reflection
    VectorCoord reflectedColor = emitLight(reflectedRay, depth + 1);
    
    // The overall color is a combination of the color computed from the Phong model and the color of the reflection
    VectorCoord color = localColor + reflectedColor * 0.8;
    
    return color;
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
	if (vertices != nullptr && normals != nullptr && numTriangles < MAX_TRIANGLES) {
		for (int i = 0; i < 3; i++) {
			GzVertex v(vertices[i], normals[i]);
			triangles[numTriangles] = GzTriangle(v, v, v);
			numTriangles++;
		}
	}





	//float d= 1.0/tan((m_camera.FOV / 2) * (PI / 180));
	//if (vertices != NULL) {
	//	// Sort vertices by y
	//	std::vector<std::vector<float>> verticesVec, normalVec, UVlistVec;
	//	std::vector<std::vector<float>> outputVec(3, std::vector<float>(3));
	//	// Add each GzCoord to the vector
	//	for (int i = 0; i < 3; ++i)
	//	{
	//		verticesVec.push_back(std::vector<float>(vertices[i], vertices[i] + 3));
	//		normalVec.push_back(std::vector<float>(normals[i], normals[i] + 3));
	//		UVlistVec.push_back(std::vector<float>(uvlist[i], uvlist[i] + 2));
	//	}
	//	for (int count = matlevel; count >= 0; count--) {
	//		GzMatrix& matrix = Ximage[count];
	//		GzMatrix& Xn = Xnorm[count];
	//		for (int h = 0; h < 3; ++h) {
	//			std::vector<float> resultVec0(4);  // ������� ��ʼֵΪ0
	//			std::vector<float> resultVec1(4);  // ������� ��ʼֵΪ0
	//			verticesVec[h].push_back(1.0);  // ��չ��4D����
	//			normalVec[h].push_back(1.0);
	//			for (int j = 0; j < 4; ++j) {
	//				for (int k = 0; k < 4; ++k) {
	//					resultVec0[j] += matrix[j][k] * verticesVec[h][k];
	//					resultVec1[j] += Xn[j][k] * normalVec[h][k];
	//				}
	//			}
	//			verticesVec[h].pop_back();
	//			normalVec[h].pop_back();
	//			verticesVec[h][0] = resultVec0[0] / resultVec0[3];
	//			verticesVec[h][1] = resultVec0[1] / resultVec0[3];
	//			verticesVec[h][2] = resultVec0[2] / resultVec0[3];
	//			normalVec[h][0] = resultVec1[0] / resultVec1[3];
	//			normalVec[h][1] = resultVec1[1] / resultVec1[3];
	//			normalVec[h][2] = resultVec1[2] / resultVec1[3];
	//			if (count == 2 && verticesVec[h][2] < 0) {
	//				return GZ_FAILURE;
	//			}
	//		}
	//	}



	//	// Pair each vertex with its corresponding index
	//	std::vector<std::pair<std::vector<float>, int>> indexedVertices;
	//	for (size_t i = 0; i < verticesVec.size(); ++i)
	//		indexedVertices.push_back({ verticesVec[i], i });

	//	// Sort vertices by y, preserving original indices
	//	std::sort(indexedVertices.begin(), indexedVertices.end(), [](const auto& a, const auto& b) {return a.first[Y] < b.first[Y]; });

	//	// Rearrange verticesVec and normalVec in the sorted order
	//	std::vector<std::vector<float>> sortedVerticesVec, sortedNormalVec,sortedUVlistVec;
	//	int firstIndex = -1, indexCount = 0;
	//	for (const auto& pair : indexedVertices)
	//	{
	//		//firstIndex++;
	//		if (pair.second == 0) {
	//			firstIndex = indexCount;
	//		}
	//		sortedVerticesVec.push_back(pair.first);
	//		sortedNormalVec.push_back(normalVec[pair.second]);
	//		sortedUVlistVec.push_back(UVlistVec[pair.second]);
	//		indexCount++;

	//	}

	//	verticesVec = sortedVerticesVec;
	//	normalVec = sortedNormalVec;
	//	UVlistVec = sortedUVlistVec;
	//	for (int i = 0; i < 3; i++) {
	//		for (int j = 0; j < 3; j++) {
	//			vertices[i][j] = verticesVec[i][j];
	//			normals[i][j] = normalVec[i][j];
	//			
	//		}
	//		for (int j = 0; j < 2; j++) {
	//			
	//			uvlist[i][j] = UVlistVec[i][j];
	//		}


	//	}
	//	GzColor textureColor;

	//	for (int i = 0; i < 3; ++i) {
	//		float z = vertices[i][2];
	//		float VZ = z / (INT_MAX - z);
	//		float adjustZ = (VZ + 1);
	//		UVlistVec[i][0] = UVlistVec[i][0] / adjustZ; // (10)
	//		UVlistVec[i][1] = UVlistVec[i][1] / adjustZ;
	//	}
	//	GzCoord viewDirection = { 0.0f, 0.0f, -1.0f };  // The direction of the eye
	//	GzColor* vecColor = new GzColor[3];  // Reset the color for each light
	//	for (int i = 0; i < 3; ++i) {  // Loop for each vertex
	//
	//		GzCoord currNormalVec;
	//		for (int j = 0; j < 3; ++j) {
	//			vecColor[i][j] = 0.0f;
	//		}
	//		for (int j = 0; j < 3; j++) {
	//			currNormalVec[j] = normals[i][j];
	//		}
	//		//GzColor color = { 0.0f, 0.0f, 0.0f };  // Reset the color for each light
	//		for (int j = 0; j < numlights; ++j) {
	//			GzLight light = lights[j];

	//			float dotProductNormLight = dot(currNormalVec, light.direction);
	//			float dotProductNormView = dot(currNormalVec, viewDirection);
	//			if (dotProductNormLight < 0 && dotProductNormView < 0) {
	//				// flip normal vector if dot product under zero
	//				currNormalVec[0] = -currNormalVec[0];
	//				currNormalVec[1] = -currNormalVec[1];
	//				currNormalVec[2] = -currNormalVec[2];
	//				dotProductNormLight = -dotProductNormLight;
	//			}
	//			else if (dotProductNormLight * dotProductNormView < 0) {
	//				// light and eye are on different sides of the surface, skip this light
	//				continue;
	//			}
	//			// Calculate the diffuse light
	//			/*float dotProduct = max(0.0f, dot(currNormalVec, light.direction));*/
	//			GzColor diffuseLight;
	//			// Calculate the diffuse light
	//			for (int k = 0; k < 3; k++) {
	//				diffuseLight[k] = Kd[k] * light.color[k] * dotProductNormLight;  // Apply Kd, color and dotProduct element-wise
	//			}

	//			// Calculate the specular light
	//			GzCoord reflection = { 0.0f, 0.0f, 0.0f };
	//			for (int k = 0; k < 3; k++) {
	//				reflection[k] = 2 * dotProductNormLight * currNormalVec[k] - light.direction[k];
	//			}
	//			VectorCoord vecReflection(3);
	//			vecReflection = normalize(reflection);
	//			float viewerDotReflection = max(0.0f, dot(vecReflection, viewDirection));
	//			GzColor specularLight = { 0.0f, 0.0f, 0.0f };
	//			for (int k = 0; k < 3; k++) {
	//				specularLight[k] = Ks[k] * light.color[k] * pow(viewerDotReflection, spec);  // Apply Ks, color and pow(...) operation element-wise
	//			}
	//			// Add the diffuse and specular light to the total color
	//			for (int k = 0; k < 3; k++) {
	//				vecColor[i][k] += diffuseLight[k] + specularLight[k];  // Add diffuseLight and specularLight to color element-wise
	//			}
	//		}

	//		 /*Now 'color' contains the color of the light at the current vertex
	//		 Do something with 'color', like storing it in an array for later use, etc.*/
	//		for (int k = 0; k < 3; k++) {
	//			vecColor[i][k] += Ka[k] * ambientlight.color[k];  // Multiply Ka and color element-wise
	//		}
	//	}
	//	// Scanline



	//	for (float scanline = std::ceil(verticesVec[0][Y]); scanline <= verticesVec[2][Y]; scanline += 1.0f) {
	//		// Interpolate x,z for the scanline
	//		float x1, x2, z1, z2;
	//		float t;
	//		VectorCoord color1(3), color2(3);
	//		VectorCoord normal1(3), normal2(3);
	//		VectorCoord uvlist1(2), uvlist2(2);
	//		// Interpolate between y0-y1 and y0-y2
	//		if (scanline <= verticesVec[1][Y]) {

	//			if ((verticesVec[1][Y] == verticesVec[0][Y])) {
	//				t = 0;
	//			}
	//			else
	//				t = (scanline - verticesVec[0][Y]) / (verticesVec[1][Y] - verticesVec[0][Y]);
	//			x1 = interpolate(verticesVec[0][X], verticesVec[1][X], t);
	//			z1 = interpolate(verticesVec[0][Z], verticesVec[1][Z], t);
	//			uvlist1= interpolateTextureColor(UVlistVec[0], UVlistVec[1], t);
	//			/*color1 = interpolateColor(vecColor[0], vecColor[1], t);*/
	//			normal1 = interpolateColor(normalVec[0], normalVec[1], t);
	//			if ((verticesVec[2][Y] == verticesVec[0][Y])) {
	//				t = 0;
	//			}
	//			else
	//				t = (scanline - verticesVec[0][Y]) / (verticesVec[2][Y] - verticesVec[0][Y]);

	//			/*color2 = interpolateColor(vecColor[0], vecColor[2], t);*/
	//			normal2 = interpolateColor(normalVec[0], normalVec[2], t);
	//			uvlist2 = interpolateTextureColor(UVlistVec[0], UVlistVec[2], t);
	//			x2 = interpolate(verticesVec[0][X], verticesVec[2][X], t);
	//			z2 = interpolate(verticesVec[0][Z], verticesVec[2][Z], t);
	//		}
	//		else {
	//			// Interpolate between y1-y2 and y0-y2

	//			if ((verticesVec[2][Y] == verticesVec[1][Y])) {
	//				t = 0;
	//			}
	//			else
	//				t = (scanline - verticesVec[1][Y]) / (verticesVec[2][Y] - verticesVec[1][Y]);
	//			x1 = interpolate(verticesVec[1][X], verticesVec[2][X], t);
	//			/*color1 = interpolateColor(vecColor[1], vecColor[2], t);*/
	//			normal1 = interpolateColor(normalVec[1], normalVec[2], t);
	//			uvlist1 = interpolateTextureColor(UVlistVec[1], UVlistVec[2], t);
	//			z1 = interpolate(verticesVec[1][Z], verticesVec[2][Z], t);
	//			if ((verticesVec[2][Y] == verticesVec[0][Y])) {
	//				t = 0;
	//			}
	//			else
	//				t = (scanline - verticesVec[0][Y]) / (verticesVec[2][Y] - verticesVec[0][Y]);
	//			x2 = interpolate(verticesVec[0][X], verticesVec[2][X], t);
	//			z2 = interpolate(verticesVec[0][Z], verticesVec[2][Z], t);
	//			uvlist2 = interpolateTextureColor(UVlistVec[0], UVlistVec[2], t);
	//			normal2 = interpolateColor(normalVec[0], normalVec[2], t);
	//			/*color2 = interpolateColor(vecColor[0], vecColor[2], t);*/
	//		}

	//		// Draw horizontal line for this scanline
	//		if (x1 > x2) {
	//			std::swap(x1, x2); std::swap(z1, z2); std::swap(color1, color2); std::swap(normal1, normal2); std::swap(uvlist1, uvlist2);
	//		}

	//		for (int x = std::ceil(x1); x <= std::floor(x2); ++x) {
	//			float z = interpolate(z1, z2, (x - x1) / (x2 - x1));

	//			// Calculate the color of the current pixel
	//			float t = (x - x1) / (x2 - x1);
	//			/*VectorCoord currColor = interpolateColor(color1, color2, t);*/
	//			VectorCoord currNormalVec = interpolateColor(normal1, normal2, t);
	//			VectorCoord curruvlist= interpolateTextureColor(uvlist1, uvlist2, t);
	//			
	//				
	//			float VZ = z / (INT_MAX - z);
	//			float adjustZ = (VZ + 1);
	//			curruvlist[0] = curruvlist[0] * adjustZ; // (10)
	//			curruvlist[1] = curruvlist[1] * adjustZ;
	//			if (interp_mode == GZ_FLAT) {
	//				GzPut(x, scanline, ctoi(vecColor[firstIndex][RED]), ctoi(vecColor[firstIndex][GREEN]), ctoi(vecColor[firstIndex][BLUE]), 1, z);
	//			}
	//			else {

	//				VectorCoord normalColor(3);
	//				GzColor textureColor;

	//				tex_fun(curruvlist[0], curruvlist[1], textureColor);

	//				for (int j = 0; j < 3; ++j) {
	//					normalColor[j] = 0.0f;
	//				}

	//				//GzColor color = { 0.0f, 0.0f, 0.0f };  // Reset the color for each light
	//				for (int j = 0; j < numlights; ++j) {
	//					GzLight light = lights[j];


	//					float dotProductNormLight = dot(currNormalVec, light.direction);
	//					float dotProductNormView = dot(currNormalVec, viewDirection);
	//					if (dotProductNormLight < 0 && dotProductNormView < 0) {
	//						// flip normal vector if dot product under zero
	//						currNormalVec[0] = -currNormalVec[0];
	//						currNormalVec[1] = -currNormalVec[1];
	//						currNormalVec[2] = -currNormalVec[2];
	//						dotProductNormLight = -dotProductNormLight;
	//						dotProductNormView = -dotProductNormView;
	//					}
	//					else if (dotProductNormLight * dotProductNormView < 0) {
	//						// light and eye are on different sides of the surface, skip this light
	//						continue;
	//					}
	//					// Calculate the diffuse light
	//					/*float dotProduct = max(0.0f, dot(currNormalVec, light.direction));*/
	//					GzColor diffuseLight;
	//					// Calculate the diffuse light
	//					for (int k = 0; k < 3; k++) {
	//						diffuseLight[k] = textureColor[k] * light.color[k] * dotProductNormLight;  // Apply Kd, color and dotProduct element-wise
	//					}

	//					// Calculate the specular light
	//					GzCoord reflection = { 0.0f, 0.0f, 0.0f };
	//					for (int k = 0; k < 3; k++) {
	//						reflection[k] = 2 * dotProductNormLight * currNormalVec[k] - light.direction[k];
	//					}
	//					VectorCoord vecReflection(3);
	//					vecReflection = normalize(reflection);
	//					float viewerDotReflection = max(0.0f, dot(vecReflection, viewDirection));
	//					GzColor specularLight = { 0.0f, 0.0f, 0.0f };
	//					for (int k = 0; k < 3; k++) {
	//						if (interp_mode == GZ_NORMALS ) {
	//							/*GzPut(x, scanline, ctoi(currColor[RED]), ctoi(currColor[GREEN]), ctoi(currColor[BLUE]), 1, z);*/
	//							specularLight[k] = Ks[k] * light.color[k] * pow(viewerDotReflection, spec);  // Apply Ks, color and pow(...) operation element-wise
	//						}
	//						else if (interp_mode == GZ_COLOR) {

	//							specularLight[k] = textureColor[k] * light.color[k] * pow(viewerDotReflection, spec);  // Apply Ks, color and pow(...) operation element-wise
	//						}
	//						
	//					}
	//					// Add the diffuse and specular light to the total color
	//					for (int k = 0; k < 3; k++) {
	//						normalColor[k] += diffuseLight[k] + specularLight[k];  // Add diffuseLight and specularLight to color element-wise
	//					}
	//				}

	//				// Now 'color' contains the color of the light at the current vertex
	//				// Do something with 'color', like storing it in an array for later use, etc.
	//				for (int k = 0; k < 3; k++) {
	//					normalColor[k] += textureColor[k] * ambientlight.color[k];  // Multiply Ka and color element-wise
	//				}
	//				GzPut(x, scanline, ctoi(normalColor[RED]), ctoi(normalColor[GREEN]), ctoi(normalColor[BLUE]), 1, z);
	//			}
	//			

	//		


	//		}
	//	}
	//}

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
 * Given triangle vertex position a, b, c; check if point p within the triangle(assume they already in the same plane)
 *
 * @param p Point P
 * @param a Triangle vertex position A
 * @param b Triangle vertex position B
 * @param c Triangle vertex position C
 * @param n Triangle plane normal vector
 * @return A boolean indiciating if the point p within triangle abc
 */
template<typename T1, typename T2, typename T3, typename T4, typename T5>
bool IsPointWithinTriangle(T1 p[3], T2 a[3], T3 b[3], T4 c[3], T5 n[3])
{
	double AB[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
	double AP[3] = { p[0] - a[0], p[1] - a[1], p[2] - a[2] };
	bool test1 = (dotArray(crossArray(AB, AP), n) >= 0);

	double BC[3] = { c[0] - b[0], c[1] - b[1], c[2] - b[2] };
	double BP[3] = { p[0] - b[0], p[1] - b[1], p[2] - b[2] };
	bool test2 = (dotArray(crossArray(BC, BP), n) >= 0);

	double CA[3] = { a[0] - c[0], a[1] - c[1], a[2] - c[2] };
	double CP[3] = { p[0] - c[0], p[1] - c[1], p[2] - c[2] };
	bool test3 = (dotArray(crossArray(CA, CP), n) >= 0);

	return test1 && test2 && test3;
}

/**
 * Given a light, return if the light is colliding with any triangle
 *
 * @param light The input light for collision detection
 * @param index The output index of first colliding triangle
 * @return A boolean indiciating if the light is colliding with any triangle
 */
bool GzRender::GzCollisionWithTriange(GzLight light, int &index)
{
	index = -1;
	double firstIntersectPos[3];
	for (int i = 0; i < triangleNum; i++)
	{
		double currIntersectPos[3];
		if (GzCollisionWithSpecificTriangle(light, triangles[i], currIntersectPos))
		{
			// If intersects, check if other triangle has collided with the light yet
			if (index == -1)
			{
				// Light not collide with other triangles yet
				index = i;
				firstIntersectPos[0] = currIntersectPos[0];
				firstIntersectPos[1] = currIntersectPos[1];
				firstIntersectPos[2] = currIntersectPos[2];
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
bool GzRender::GzCollisionWithSpecificTriangle(GzLight light, GzTriangle triangle, double intersectPos[3])
{
	double e1[3] = {triangle.v[1].position[0] - triangle.v[0].position[0],
					triangle.v[1].position[1] - triangle.v[0].position[1],
					triangle.v[1].position[2] - triangle.v[0].position[2] };
	double e2[3] = {triangle.v[2].position[0] - triangle.v[0].position[0],
					triangle.v[2].position[1] - triangle.v[0].position[1],
					triangle.v[2].position[2] - triangle.v[0].position[2] };

	// Normal to triangle plane
	float e1e2Norm[3];
	crossArray(e1, e2, e1e2Norm);

	// If flat triangle normal perpendicular to light, then no intersection in terms of direction
	if (abs(dot(e1e2Norm, light.direction) < 0.0001))
	{
		return false;
	}

	// Compute intersection point(with triangle's plane)
	// d = n dot x
	// t = (d - n dot p) / (n dot d)
	float d = dotArray(e1e2Norm, triangle.v[0].position);
	double t = (d - dotArray(e1e2Norm, light.origin)) / dotArray(e1e2Norm, light.direction);
	intersectPos[0] = light.origin[0] + t * light.direction[0];
	intersectPos[1] = light.origin[1] + t * light.direction[1];
	intersectPos[2] = light.origin[2] + t * light.direction[2];

	// Check if the point is within triangle
	return IsPointWithinTriangle(intersectPos, triangle.v[0].position, triangle.v[1].position, triangle.v[2].position, e1e2Norm);
}

/**
 * Given a light and a triangle(need to be intersected beforehand), get its reflective and refractive light
 *
 * @param light The input light
 * @param triangle The input triangle
 * @param reflectLight The output reflective light
 * @param refractLight The output refractive light
 */
void GzRender::FresnelReflection(GzLight light, GzTriangle triangle, GzLight reflectLight, GzLight refractLight)
{
	double refractIndex = 1.5; // Self-defined to 1.5 because the storing location of refractive index undecided yet


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
bool RayIntersectsSphere(const double origin[3], const double direction[3], const double center[3], double radius, double &t) {
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
bool GzRender::GzCollisionWithSphere(GzLight light, int &index) {
    index = -1;
    double closest_t = std::numeric_limits<double>::max();
    double t;
    for (int i = 0; i < sphereNum; i++) {
        if (RayIntersectsSphere(light.origin, light.direction, spheres[i].center, spheres[i].radius, t)) {
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
void GzRender::CalculateSphereReflectionAndRefraction(GzLight light, GzSphere sphere, const double intersectionPoint[3], GzLight &reflectLight, GzLight &refractLight) {
    // Calculate normal at the intersection point
    double normal[3] = {
        (intersectionPoint[0] - sphere.center[0]) / sphere.radius,
        (intersectionPoint[1] - sphere.center[1]) / sphere.radius,
        (intersectionPoint[2] - sphere.center[2]) / sphere.radius
    };

    // Calculate reflection vector
    double dot_ln = dotArray(light.direction, normal);
    reflectLight.direction[0] = light.direction[0] - 2 * dot_ln * normal[0];
    reflectLight.direction[1] = light.direction[1] - 2 * dot_ln * normal[1];
    reflectLight.direction[2] = light.direction[2] - 2 * dot_ln * normal[2];

    // TODO: refraction
}